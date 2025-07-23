import os
import io
import base64
import asyncio
import random
import numpy as np
from PIL import Image
from ollama import chat
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from torchvision.transforms.functional import to_pil_image
import torchvision.models.segmentation as segmentation
import clip
from lpips import LPIPS
from diffusers import StableDiffusionImg2ImgPipeline
from datasets import load_dataset
from pydantic import BaseModel, Field, ValidationError, create_model
import re

SD_MODEL_NAME = "stabilityai/stable-diffusion-2-1-base"
NEG_PROMPT_MODEL = "llama3.1"
EVAL_MODEL = "mistral-small3.1"
RUNS_PER_IMAGE = 100
HF_DATASET = "SargeZT/coco-stuff-captioned"
SAVE_BAD_IMAGES = False

N_INCREASE_THRESHOLD = 0.2
OUTPUT_BASE_PATH = f"../test_outputs_nway_{EVAL_MODEL}_{N_INCREASE_THRESHOLD}"
os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

INITIAL_SAMPLE_SIZE = 100

def count_image_lines(file_path):
    pattern = re.compile(r'^Image \d+: .+')
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if pattern.match(line.strip()):
                count += 1
    print("[DEBUG] Counted lines in file:", file_path, "Count:", count)
    return count

SAMPLE_SIZE = max(INITIAL_SAMPLE_SIZE - count_image_lines(os.path.join(OUTPUT_BASE_PATH, "results_summary.txt")), 0)
print("[DEBUG] Sample size left for processing:", SAMPLE_SIZE)

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    SD_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
sd_pipeline.enable_model_cpu_offload()
segmentation_model = segmentation.deeplabv3_resnet101(pretrained=True).to(device)
segmentation_model.eval()
lpips_model = LPIPS(net='vgg').to(device)

class SiameseSemanticNetwork(torch.nn.Module):
    def __init__(self, image_embed_dim=512, text_embed_dim=1024, hidden_dim=1024, output_dim=1024):
        super().__init__()
        self.common_branch = torch.nn.Sequential(
            torch.nn.Linear(image_embed_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.BatchNorm1d(output_dim)
        )
        self.distinctive_branch = torch.nn.Sequential(
            torch.nn.Linear(image_embed_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
            torch.nn.BatchNorm1d(output_dim)
        )
        self.text_projection = torch.nn.Linear(text_embed_dim, output_dim)
        self.semantic_bn = torch.nn.BatchNorm1d(output_dim, affine=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, image_embed, text_embed=None, mode='both'):
        common_features = self.common_branch(image_embed)
        distinctive_features = self.distinctive_branch(image_embed)
        if mode == 'common':
            return common_features
        elif mode == 'distinctive':
            return distinctive_features
        else:
            if text_embed is not None:
                projected_text = self.text_projection(text_embed)
                distinctive_features = self.semantic_bn(distinctive_features)
                semantic_weight = torch.sigmoid(projected_text)
                modulated_distinctive = distinctive_features * semantic_weight
                combined_features = common_features + modulated_distinctive
                return combined_features, common_features, modulated_distinctive
            return common_features + distinctive_features, common_features, distinctive_features

siamese_network = SiameseSemanticNetwork(
    image_embed_dim=512,
    text_embed_dim=1024,
    hidden_dim=1024,
    output_dim=1024
).to(device)
siamese_network.eval()

class SemanticLayoutGenerator(torch.nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=512):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1024),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, text_embed, image_embed):
        combined = torch.cat([text_embed, image_embed], dim=1)
        encoded = self.encoder(combined)
        batch_size = encoded.shape[0]
        reshaped = encoded.view(batch_size, 256, 2, 2)
        layout = self.decoder(reshaped)
        return layout

layout_generator = SemanticLayoutGenerator().to(device)
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

def encode_image(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def concatenate_images_pil(images: list[Image.Image]) -> Image.Image:
    
    min_height = min(img.height for img in images)
    resized = [img.resize((int(img.width * min_height / img.height), min_height), Image.LANCZOS) for img in images]
    total_width = sum(img.width for img in resized)
    concat_img = Image.new("RGB", (total_width, min_height))
    x = 0
    for idx, img in enumerate(resized):
        concat_img.paste(img, (x, 0))
        x += img.width
        img.save(os.path.join(OUTPUT_BASE_PATH, f"img{idx}.jpg"))
    concat_img.save(os.path.join(OUTPUT_BASE_PATH, "concat.jpg"))
    
    return concat_img

def letter_options(n):
    return [chr(ord('A') + i) for i in range(n)]

def generate_semantic_layout(image, text_embed, image_embed):
    with torch.no_grad():
        layout = layout_generator(text_embed.float(), image_embed.float())
        layout = layout.squeeze(0).squeeze(0).cpu().numpy()
        layout = Image.fromarray((layout * 255).astype(np.uint8)).resize(image.size)
    preprocess = Compose([
        Resize((520, 520)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = segmentation_model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()
    segmentation_mask = Image.fromarray(output_predictions).resize(image.size)
    layout_array = np.array(layout)
    segmentation_array = np.array(segmentation_mask)
    enhanced_layout = layout_array * (segmentation_array > 0)
    enhanced_layout = Image.fromarray(enhanced_layout.astype(np.uint8))
    return enhanced_layout

class NegativePromptModel(BaseModel):
    negative_prompt: str = Field(...)

async def get_negative_prompt_ollama(image_caption):
    while True:
        try:
            prompt = (
                f"Given the following image description: \"{image_caption}\", "
                "generate a detailed negative prompt for Stable Diffusion that will make an image of this type look much worse in quality. "
                "Focus on flaws, defects, poor condition, and use context-specific terms (e.g., wilted for flowers, worn out for shoes, etc). "
            )
            messages = [
                {"role": "system", "content": "You are an expert at crafting negative prompts for Stable Diffusion."},
                {"role": "user", "content": prompt}
            ]
            response = await asyncio.to_thread(
                chat,
                model=NEG_PROMPT_MODEL,
                messages=messages,
                options={"temperature": 0.1},
                format=NegativePromptModel.model_json_schema(),
            )
            text = response['message']['content'].strip()
            return NegativePromptModel.model_validate_json(text).model_dump()
        except ValidationError as ve:
            print(f"[DEBUG] Pydantic validation error: {ve}")

def semantic_disentanglement_injection(
    image, positive_text, negative_text="low quality, basic, generic",
    num_steps=20, lr=0.005, strength_values=None, guidance_scale_values=None,
    resize_size=(512, 512)
):
    resize_transform = Resize(resize_size)
    image = resize_transform(image)
    clip_image = clip_preprocess(image).unsqueeze(0).to(device)
    sd_image = ToTensor()(image).unsqueeze(0).to(device).half()

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_embed = clip_model.encode_image(clip_image.float())

    with torch.no_grad():
        positive_text_embed = sd_pipeline.text_encoder(
            sd_pipeline.tokenizer(
                [positive_text],
                padding="max_length",
                truncation=True,
                max_length=sd_pipeline.tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids.to(device)
        ).last_hidden_state.half()
        negative_text_embed = sd_pipeline.text_encoder(
            sd_pipeline.tokenizer(
                [negative_text],
                padding="max_length",
                truncation=True,
                max_length=sd_pipeline.tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids.to(device)
        ).last_hidden_state.half()

    semantic_layout = generate_semantic_layout(
        image,
        positive_text_embed.mean(dim=1),
        image_embed
    )
    layout_tensor = ToTensor()(semantic_layout).unsqueeze(0).to(device)
    optimized_embed = image_embed.clone().detach().float().requires_grad_(True)
    optimizer = torch.optim.Adam([optimized_embed], lr=lr)
    results = []
    for strength in strength_values:
        for guidance_scale in guidance_scale_values:
            for _ in range(num_steps):
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    combined_features, _, distinctive_features = siamese_network(
                        optimized_embed,
                        positive_text_embed.mean(dim=1).float()
                    )
                    layout_guided_features = combined_features * layout_tensor.mean(dim=[2, 3]).float()
                    fused_embed = torch.tanh(layout_guided_features.unsqueeze(1))
                    fused_embed = fused_embed.repeat(1, 77, 1)
                    updated_image = sd_pipeline(
                        prompt_embeds=fused_embed,
                        negative_prompt_embeds=negative_text_embed,
                        image=to_pil_image(sd_image[0].cpu()),
                        strength=strength,
                        guidance_scale=guidance_scale
                    ).images[0]
                    updated_tensor = ToTensor()(updated_image).unsqueeze(0).to(device).float()
                    perc_loss = lpips_model(sd_image.float() * 2 - 1, updated_tensor * 2 - 1).mean()
                    with torch.no_grad():
                        updated_embed = clip_model.encode_image(clip_preprocess(updated_image).unsqueeze(0).to(device))
                        positive_text_clip = clip_model.encode_text(clip.tokenize([positive_text]).to(device))
                    semantic_loss = 1.0 - F.cosine_similarity(updated_embed, positive_text_clip).mean()
                    distinctive_loss = F.mse_loss(
                        siamese_network(updated_embed, mode='distinctive'),
                        distinctive_features
                    )
                    total_loss = perc_loss + 0.5 * semantic_loss + 0.3 * distinctive_loss
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(optimized_embed, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(optimized_embed, 1.0)
                    optimizer.step()
            with torch.no_grad():
                combined_features, _, _ = siamese_network(
                    optimized_embed,
                    positive_text_embed.mean(dim=1).float()
                )
                layout_guided_features = combined_features * layout_tensor.mean(dim=[2, 3]).float()
                fused_embed = torch.tanh(layout_guided_features.unsqueeze(1))
                result = sd_pipeline(
                    prompt_embeds=fused_embed.repeat(1, 77, 1),
                    negative_prompt_embeds=negative_text_embed,
                    image=to_pil_image(sd_image[0].cpu()),
                    strength=strength,
                    guidance_scale=guidance_scale
                ).images[0]
                layout_overlay = semantic_layout.copy().convert('RGBA')
                layout_overlay.putalpha(128)
                result_with_layout = result.copy()
                result_with_layout.paste(layout_overlay, (0, 0), layout_overlay)
                results.append((strength, guidance_scale, result.convert('RGB'), result_with_layout))
    return results

def make_action_response_model(options_str):
    
    pattern = f"^({options_str})$"
    
    return create_model(
        'ActionResponse',
        explanation=(str, ...),
        action=(str, Field(..., pattern=pattern)),
    )

async def query_ollama_eval_nway(concat_image_b64: str, prompt: str, model: str, n: int):
    options = letter_options(n)
    options_str = "|".join(options)  
    options_display = "/".join(options)  
    ActionResponse = make_action_response_model(options_str)
    system_msg = (
        f"You are a helpful assistant that compares {n} images and selects the best one. "
        f"Respond with one of {options_display}. Explain your reasoning. "
        f"Respond in this JSON format: {{\"action\": \"{options_display}\", \"explanation\": \"...\"}}"
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt, "images": [concat_image_b64]}
    ]
    try:
        response = await asyncio.to_thread(
            chat,
            model=model,
            messages=messages,
            options={"temperature": 0.1, "num_predict": 500},
            format=ActionResponse.model_json_schema(),
        )
        text = response['message']['content']
        
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = ActionResponse.model_validate_json(match.group(0)).model_dump()
                action = data.get("action", "").strip().upper()
                print("[DEBUG] Action:", action, "Explanation:", data.get("explanation", ""))
                if action in options:
                    return action
            except ValidationError as ve:
                print(f"[DEBUG] Pydantic validation error: {ve}")
        
        for letter in options:
            if text.strip().startswith(letter):
                return letter
        return "ERROR"
    except Exception as e:
        print(f"Ollama LLM eval error: {e}")
        return "ERROR"

MAX_ITER = 20
OPTIM_STEPS = 20
INIT_STRENGTH = 0.5
INIT_CFG = 7.5
STRENGTH_STEP = 0.1
CFG_STEP = 1.0
STRENGTH_RANGE = (0.3, 0.8)
CFG_RANGE = (2.0, 12.0)

async def optimize_case(
    bad_image: Image.Image,
    positive_text: str,
    negative_text: str,
    eval_model: str,
    image_id: str,
    normal_images: list[Image.Image],
    n_variations: int
):
    strength = INIT_STRENGTH
    cfg = INIT_CFG
    best_majority = 0.0
    best_image_path = None
    best_s = strength
    best_c = cfg
    best_image = bad_image
    concat_img = None

    for iteration in range(MAX_ITER):
        print("-" * 20)
        print(f"Iteration {iteration + 1}/{MAX_ITER}: Strength={strength:.2f}, CFG={cfg:.2f}")
        print("-" * 20)
        
        results = semantic_disentanglement_injection(
            image=bad_image,
            positive_text=positive_text,
            negative_text=negative_text,
            num_steps=OPTIM_STEPS,
            lr=0.05,
            strength_values=[strength],
            guidance_scale_values=[cfg]
        )
        _, _, gen_image, _ = results[0]
        gen_image_path = os.path.join(
            OUTPUT_BASE_PATH, f"{image_id}_opt_iter_{iteration+1}_s{strength:.2f}_c{cfg:.2f}.jpg"
        )

        nway_votes = 0
        for _ in range(RUNS_PER_IMAGE):
            images = normal_images.copy()
            insert_pos = random.randint(0, n_variations - 1)
            images.insert(insert_pos, gen_image)
            concat_img = concatenate_images_pil(images)
            options = letter_options(n_variations)
            prompt_text = (
                f"You see {n_variations} images (labeled {'/'.join(options)} from left to right). "
                f"Which image is the best? Respond with one of {'/'.join(options)}."
            )
            concat_img_b64 = encode_image(concat_img)
            chosen_option = await query_ollama_eval_nway(concat_img_b64, prompt_text, eval_model, n_variations)
            if chosen_option == options[insert_pos]:
                nway_votes += 1
            print(f"  [DEBUG] n-way eval run: optimized image at {options[insert_pos]}, LLM chose {chosen_option}")

        percent_chosen = nway_votes / RUNS_PER_IMAGE
        print(f"  [DEBUG] Iteration {iteration+1}: optimized image chosen {nway_votes}/{RUNS_PER_IMAGE} ({percent_chosen:.2%})")

        if percent_chosen  >= (1 / n_variations) + N_INCREASE_THRESHOLD:
            best_majority = percent_chosen  
            best_image_path = gen_image_path
            best_s = strength
            best_c = cfg
            best_image = gen_image
            break

        elif percent_chosen  > best_majority:
            best_majority = percent_chosen 
            best_image_path = gen_image_path
            best_s = strength
            best_c = cfg
            best_image = gen_image
        print("\n\nGenerated Image Path:", gen_image_path, "\n\n")
        
        candidates = []
        for d_strength in [-STRENGTH_STEP, 0, STRENGTH_STEP]:
            for d_cfg in [-CFG_STEP, 0, CFG_STEP]:
                if d_strength == 0 and d_cfg == 0:
                    continue
                s_new = np.clip(strength + d_strength, *STRENGTH_RANGE)
                c_new = np.clip(cfg + d_cfg, *CFG_RANGE)
                candidates.append((s_new, c_new))

        candidate_scores = []
        for s, c in candidates:
            results = semantic_disentanglement_injection(
                image=bad_image,
                positive_text=positive_text,
                negative_text=negative_text,
                num_steps=OPTIM_STEPS,
                lr=0.05,
                strength_values=[s],
                guidance_scale_values=[c]
            )
            _, _, cand_image, _ = results[0]
            nway_votes = 0
            for _ in range(RUNS_PER_IMAGE):
                images = normal_images.copy()
                insert_pos = random.randint(0, n_variations - 1)
                images.insert(insert_pos, cand_image)
                concat_img = concatenate_images_pil(images)
                options = letter_options(n_variations)
                prompt_text = (
                    f"You see {n_variations} images (labeled {'/'.join(options)} from left to right). "
                    f"Which image is the best? Respond with one of {'/'.join(options)}."
                )
                concat_img_b64 = encode_image(concat_img)
                chosen_option = await query_ollama_eval_nway(concat_img_b64, prompt_text, eval_model, n_variations)
                if chosen_option == options[insert_pos]:
                    nway_votes += 1
                print(f"  [DEBUG] n-way eval run: optimized image at {options[insert_pos]}, LLM chose {chosen_option}")
            percent_cand_chosen = nway_votes / RUNS_PER_IMAGE
            print(f"    [DEBUG] Neighbor (s={s:.2f}, c={c:.2f}): optimized image chosen {nway_votes}/{RUNS_PER_IMAGE} ({percent_cand_chosen:.2%})")
            candidate_scores.append((percent_cand_chosen, s, c, cand_image, None))
            if percent_cand_chosen  >= (1 / n_variations) + N_INCREASE_THRESHOLD:
                best_majority = percent_cand_chosen 
                best_s = s
                best_c = c
                print(f" [DEBUG] Found a successful neighbor, stopping: s={s:.2f}, c={c:.2f}, score={best_majority:.2%}")
                break
        if candidate_scores:
            best_score, best_s_cand, best_c_cand, best_cand_image, best_cand_path = max(candidate_scores, key=lambda x: x[0])
            if best_score   >= (1 / n_variations) + N_INCREASE_THRESHOLD:
                best_majority = best_score  
                best_s = best_s_cand
                best_c = best_c_cand
                best_image_path = best_cand_path
                print(f" [DEBUG] Found a successful neighbor, stopping: s={best_s_cand:.2f}, c={best_c_cand:.2f}, score={best_majority:.2%}")
                break
            elif best_score  > best_majority:
                strength, cfg = best_s_cand, best_c_cand
                best_majority = best_score 
                best_image = best_cand_image
                best_image_path = best_cand_path
                print(f"  [DEBUG] Found a better neighbor, but not enough to stop: s={strength:.2f}, c={cfg:.2f}, score={best_majority:.2%}")
            else:
                print("  [DEBUG] No better neighbor found, moving to next iteration.")
    if concat_img and best_image_path:
        concat_img.save(best_image_path)
        print(f"Saved optimized image to {best_image_path}")
                
    return best_majority, best_s, best_c

async def main(n_variations,
    hf_dataset=HF_DATASET,
    sample_size=SAMPLE_SIZE,
    save_bad_images=False,
    eval_model=EVAL_MODEL
):
    dataset = load_dataset(hf_dataset, split="train")
    indices = random.sample(range(len(dataset)), sample_size)
    stats = [] 
    processed_count = 0
    with open(os.path.join(OUTPUT_BASE_PATH, "results_summary.txt"), "a") as f:
        for index, idx in enumerate(indices):
            print(f"\nProcessing {index}th image index:", idx)
            print("-" * 20)
            row = dataset[idx]
            pil_image = row["image"]
            image_caption = row["caption"]

            negative_prompt = await get_negative_prompt_ollama(image_caption)
            negative_prompt = negative_prompt['negative_prompt']
            print(f"\n\nNegative prompt for image {idx}: {negative_prompt}\n\n")
            print("-" * 20)
            
            bad_image = sd_pipeline(
                prompt=negative_prompt,
                image=pil_image,
                strength=0.65,
                guidance_scale=4.0,
            ).images[0]
            if save_bad_images:
                bad_path = os.path.join(OUTPUT_BASE_PATH, f"bad_{idx}.jpg")
                bad_image.save(bad_path)
                print(f"Saved bad image to {bad_path}")

            normal_images = []
            for _ in range(n_variations - 1):
                img = sd_pipeline(
                    prompt=image_caption,
                    image=pil_image,
                    strength=0.65,
                    guidance_scale=7.5,
                ).images[0]
                normal_images.append(img)

            best_score, best_s, best_c = await optimize_case(
                bad_image, image_caption, negative_prompt, eval_model, f"img{idx}", normal_images, n_variations
            )

            processed_count += 1
            progress_str = f"Progress: {processed_count}/{INITIAL_SAMPLE_SIZE}"
            print(progress_str)
            f.write(progress_str + "\n")

            stat = {
                "idx": idx,
                "caption": image_caption,
                "negative_prompt": negative_prompt,
                "best_score": best_score,
                "best_image_path": f"{OUTPUT_BASE_PATH}/img{idx}_opt_iter_{processed_count}_s{best_s:.2f}_c{best_c:.2f}.jpg",
            }
            stats.append(stat)
            print(f"Image {idx}: Optimized image chosen {best_score:.2%} of the time.")

            
            f.write(f"Image {stat['idx']}: {stat['caption']}\n")
            f.write(f"Negative Prompt: {stat['negative_prompt']}\n")
            f.write(f"Optimized image chosen {stat['best_score']:.2%} of the time. Path: {stat['best_image_path']}\n\n")
            print("-" * 20)
        
        def count_majority(file_path):
            pattern = re.compile(r'^Optimized image chosen  \d+% .+')
            count = 0
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if int(pattern.match(line.strip())) >= (1 / n_variations) + N_INCREASE_THRESHOLD:
                        count += 1
            return count
        num_majority = count_majority(os.path.join(OUTPUT_BASE_PATH, "results_summary.txt"))
        overall = num_majority / INITIAL_SAMPLE_SIZE if INITIAL_SAMPLE_SIZE > 0 else 0.0
        overall_str = f"OVERALL: {num_majority} out of {INITIAL_SAMPLE_SIZE} ({overall:.2%}) cases achieved majority choice."
        print(overall_str)
        f.write(overall_str + "\n")
    print(f"Summary saved to {os.path.join(OUTPUT_BASE_PATH, 'results_summary.txt')}")

if __name__ == "__main__":
    asyncio.run(main(n_variations=4))  