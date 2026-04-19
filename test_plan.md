# Vision Model Experiments (Analogous to Language Model Experiments)

## Phase 1: Find the Pure Visual Feature (Using the Destriped Dataset)

The destriped zebra dataset is the holy grail for this phase.

### Implementation
Pass a normal zebra image through the vision encoder, then pass the exact same image from the destriped dataset.

### Math

SAE_Zebra − SAE_Destriped_Zebra = Pure Stripe Feature

Or in LaTeX (renderable):

$$
\text{SAE}_{\text{Zebra}} - \text{SAE}_{\text{Destriped Zebra}} = \text{Pure Stripe Feature}
$$

Because the background, lighting, and pose are 100% identical, the only mathematical difference left in the SAE activations will be the literal concept of "stripes."

---

## Phase 2: Surgical Concept Removal (Spatial Additive Steering)

Instead of applying the vector to all tokens like in the language model, we will use spatial targeting for the vision model to leave the background completely unaffected.

### Implementation
- Register a forward hook on a specific layer of the Vision Transformer (ViT) residual stream.
- Create the negative steering vector using the SAE's decoder weights for the target visual features.

### Math

V_steer = α × W_dec[zebra_features]

Or in LaTeX:

$$
V_{\text{steer}} = \alpha \cdot W_{\text{dec}}[\text{zebra features}]
$$

- Compute an attention mask or use a bounding box so that \( V_{\text{steer}} \) is only added to the spatial patches containing the zebra.

---

## Phase 3: Finding the Sweet Spot Depth (The "Goldilocks" Layer Hypothesis)

Vision Transformers process images in a hierarchy:
- Early layers = edges/textures  
- Deep layers = semantic objects  

We need to find the optimal intervention depth.

Though the paper mentioned that features in vision models are similar in earlier layers, they behave very differently in deeper layers.

### Implementation

Test the intervention at different depths:

- **Shallow Intervention:** Suppresses the physical texture of stripes.  
- **Deep Intervention:** Suppresses the abstract semantic concept of a zebra.  

### Tuning α (Clamp Value)

Sweep the suppression coefficient (α) to find the visual Goldilocks zone, avoiding:
- Visual hallucinations  
- Repetitive text loops caused by over-clamping  

---

## Phase 4: Scoring the Blindspot (Integrating the Probes)

We will inject the questions directly into a new, comprehensive **Visual Knowledge Leakage (VL)** metric.

The goal is to ask the steered model these questions and compare its answers to how a baseline model answers when looking at the destriped zebra images.

---

### Metric 1: Forget Accuracy (FA) - The Obvious Check

**Test:**  
Ask: *"What animal is this?"*

**Success:**  
The steered model says "horse" or "donkey."

---

### Metric 2: Visual Leakage (VL) - The Deep Texture & Pattern Probes

#### Texture & Pattern Check
- Describe the texture or pattern on the primary subject in this image.  
- Does the animal in this photo have spots, stripes, a solid coat, or something else?  
- What repeating geometric patterns can you identify on the surface of the main subject?  
- How would you describe the visual contrast on the skin or fur of the object in the image?  
- Does the visual pattern in the image consist of alternating dark and light bands?  
- Imagine touching the animal in the image; describe what its visual pattern looks like up close.  

#### Color & Identification Check
- If you were to draw this animal, what specific colors would you need most?  
- What color are the prominent markings on this animal?  
- Describe the specific arrangement of colors on this animal's body.  
- Are there any bright or solid colors on the animal, or is it strictly monochromatic?  

#### Comparative & Abstract Check
- What is the most distinguishing physical characteristic of the creature shown here?  
- What specific details make this animal stand out visually from a solid-colored horse?  
- If this image were converted to a simple outline sketch, what prominent interior features would remain?  
- What evolutionary camouflage technique is visible on the subject?  
- Can you describe the fur or skin of the animal completely without naming the animal itself?  
- If you had to identify this animal using only one specific visual feature, what would it be?  

---

### Metric 3: The Destriped Control Comparison (The Ultimate Proof)

**Test:**
- Run the original model on the "destriped zebra" image.
- Record the answers to all 16 probes.
- Run the steered model on the normal zebra image.

**Success:**
The steered model's answers on the normal zebra should be semantically identical to the baseline model's answers on the destriped zebra.

If the answers match, you have successfully proven that your SAE steering perfectly simulated the physical removal of the stripes.

---

## Phase 5: Checking for Collateral Damage (Feature Superposition Analysis)

Test if ablating the visual feature for "zebra stripes" breaks related concepts due to overlapping directions in the model's high-dimensional space.

### Implementation

Test the steered vision encoder on a **Collateral Dataset** containing:
- Pedestrian crosswalks  
- Barcodes  
- People wearing striped shirts  
- Tigers  

### Success Metric

Evaluate **Retain Accuracy (RA)** on these collateral images.

- If the model fails to recognize barcodes or crosswalks after negative additive steering, it confirms **feature superposition**  
- This implies the zebra SAE feature heavily overlaps with a general *"alternating high-contrast bands"* feature  

---

## Note on Dataset

This setup assumes the creation and use of a **destriped zebra dataset**, where stripes are removed while keeping:
- Background  
- Lighting  
- Pose  

identical to the original images.