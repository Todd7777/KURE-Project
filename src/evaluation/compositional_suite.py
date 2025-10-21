"""
Compositional Evaluation Suite for NRF

This module implements targeted benchmarks for:
- Attribute binding (color, size, material)
- Spatial relations (on, under, left, right)
- Explicit negation (no, without, not)

These are the failure modes where diffusion models typically struggle.
"""

import torch
from typing import List, Dict, Tuple
import json
from pathlib import Path


class CompositionalPromptGenerator:
    """
    Generate compositional prompts for evaluation.
    
    These prompts stress-test specific compositional capabilities.
    """
    
    def __init__(self):
        self.objects = ["cube", "sphere", "cylinder", "cone", "pyramid"]
        self.colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        self.sizes = ["small", "large", "tiny", "huge"]
        self.materials = ["metal", "wooden", "glass", "plastic", "rubber"]
        self.spatial_relations = ["on top of", "under", "left of", "right of", "next to", "behind", "in front of"]
    
    def generate_attribute_prompts(self, num_prompts: int = 100) -> List[Dict[str, str]]:
        """
        Generate prompts testing attribute binding.
        
        Example: "A red cube and a blue sphere"
        
        Returns:
            prompts: List of dictionaries with 'text' and 'category'
        """
        prompts = []
        
        for i in range(num_prompts):
            # Two objects with different colors
            obj1 = self.objects[i % len(self.objects)]
            obj2 = self.objects[(i + 1) % len(self.objects)]
            color1 = self.colors[i % len(self.colors)]
            color2 = self.colors[(i + 2) % len(self.colors)]
            
            text = f"A {color1} {obj1} and a {color2} {obj2}"
            
            prompts.append({
                "text": text,
                "category": "attributes",
                "subcategory": "color_binding",
                "objects": [obj1, obj2],
                "attributes": {obj1: color1, obj2: color2},
            })
        
        # Add size attributes
        for i in range(num_prompts // 2):
            obj1 = self.objects[i % len(self.objects)]
            obj2 = self.objects[(i + 1) % len(self.objects)]
            size1 = self.sizes[i % len(self.sizes)]
            size2 = self.sizes[(i + 1) % len(self.sizes)]
            
            text = f"A {size1} {obj1} and a {size2} {obj2}"
            
            prompts.append({
                "text": text,
                "category": "attributes",
                "subcategory": "size_binding",
                "objects": [obj1, obj2],
                "attributes": {obj1: size1, obj2: size2},
            })
        
        # Add material attributes
        for i in range(num_prompts // 2):
            obj = self.objects[i % len(self.objects)]
            material = self.materials[i % len(self.materials)]
            
            text = f"A {material} {obj}"
            
            prompts.append({
                "text": text,
                "category": "attributes",
                "subcategory": "material_binding",
                "objects": [obj],
                "attributes": {obj: material},
            })
        
        return prompts
    
    def generate_spatial_prompts(self, num_prompts: int = 100) -> List[Dict[str, str]]:
        """
        Generate prompts testing spatial relations.
        
        Example: "A red cube on top of a blue sphere"
        
        Returns:
            prompts: List of dictionaries with 'text' and 'category'
        """
        prompts = []
        
        for i in range(num_prompts):
            obj1 = self.objects[i % len(self.objects)]
            obj2 = self.objects[(i + 1) % len(self.objects)]
            color1 = self.colors[i % len(self.colors)]
            color2 = self.colors[(i + 2) % len(self.colors)]
            relation = self.spatial_relations[i % len(self.spatial_relations)]
            
            text = f"A {color1} {obj1} {relation} a {color2} {obj2}"
            
            prompts.append({
                "text": text,
                "category": "spatial",
                "subcategory": relation.replace(" ", "_"),
                "objects": [obj1, obj2],
                "relation": relation,
                "attributes": {obj1: color1, obj2: color2},
            })
        
        return prompts
    
    def generate_negation_prompts(self, num_prompts: int = 100) -> List[Dict[str, str]]:
        """
        Generate prompts testing explicit negation.
        
        Example: "A cube without any red color"
        
        Returns:
            prompts: List of dictionaries with 'text' and 'category'
        """
        prompts = []
        
        negation_templates = [
            "A {obj} without any {attr}",
            "A {obj}, but not {attr}",
            "No {attr} {obj}",
            "A {obj} that is not {attr}",
        ]
        
        for i in range(num_prompts):
            obj = self.objects[i % len(self.objects)]
            color = self.colors[i % len(self.colors)]
            template = negation_templates[i % len(negation_templates)]
            
            text = template.format(obj=obj, attr=color)
            
            prompts.append({
                "text": text,
                "category": "negation",
                "subcategory": "color_negation",
                "objects": [obj],
                "negated_attribute": color,
            })
        
        # Negation with multiple objects
        for i in range(num_prompts // 2):
            obj1 = self.objects[i % len(self.objects)]
            obj2 = self.objects[(i + 1) % len(self.objects)]
            
            text = f"A {obj1} but no {obj2}"
            
            prompts.append({
                "text": text,
                "category": "negation",
                "subcategory": "object_negation",
                "objects": [obj1],
                "negated_objects": [obj2],
            })
        
        return prompts
    
    def generate_complex_prompts(self, num_prompts: int = 50) -> List[Dict[str, str]]:
        """
        Generate complex prompts combining multiple compositional challenges.
        
        Example: "A small red cube on top of a large blue sphere, but no green objects"
        
        Returns:
            prompts: List of dictionaries with 'text' and 'category'
        """
        prompts = []
        
        for i in range(num_prompts):
            obj1 = self.objects[i % len(self.objects)]
            obj2 = self.objects[(i + 1) % len(self.objects)]
            size1 = self.sizes[i % len(self.sizes)]
            color1 = self.colors[i % len(self.colors)]
            color2 = self.colors[(i + 2) % len(self.colors)]
            negated_color = self.colors[(i + 3) % len(self.colors)]
            relation = self.spatial_relations[i % len(self.spatial_relations)]
            
            text = f"A {size1} {color1} {obj1} {relation} a {color2} {obj2}, but no {negated_color} objects"
            
            prompts.append({
                "text": text,
                "category": "complex",
                "subcategory": "multi_compositional",
                "objects": [obj1, obj2],
                "attributes": {obj1: [size1, color1], obj2: [color2]},
                "relation": relation,
                "negated_attribute": negated_color,
            })
        
        return prompts
    
    def generate_full_suite(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate the complete compositional evaluation suite.
        
        Returns:
            suite: Dictionary mapping category to prompts
        """
        suite = {
            "attributes": self.generate_attribute_prompts(100),
            "spatial": self.generate_spatial_prompts(100),
            "negation": self.generate_negation_prompts(100),
            "complex": self.generate_complex_prompts(50),
        }
        
        return suite
    
    def save_suite(self, path: str):
        """Save the evaluation suite to a JSON file"""
        suite = self.generate_full_suite()
        
        with open(path, "w") as f:
            json.dump(suite, f, indent=2)
        
        print(f"Saved compositional suite to {path}")
    
    def load_suite(self, path: str) -> Dict[str, List[Dict[str, str]]]:
        """Load evaluation suite from JSON file"""
        with open(path, "r") as f:
            suite = json.load(f)
        
        return suite


class CompositionalEvaluator:
    """
    Evaluate model performance on compositional prompts.
    
    Uses CLIP to verify whether generated images satisfy compositional constraints.
    """
    
    def __init__(self, clip_model, device: str = "cuda"):
        """
        Args:
            clip_model: CLIP model for verification
            device: Device to use
        """
        self.clip_model = clip_model
        self.device = device
    
    def verify_attribute_binding(
        self,
        image: torch.Tensor,
        prompt_data: Dict[str, any],
    ) -> float:
        """
        Verify if image correctly binds attributes to objects.
        
        Args:
            image: Generated image
            prompt_data: Prompt metadata
            
        Returns:
            score: Verification score [0, 1]
        """
        # Create verification prompts
        objects = prompt_data["objects"]
        attributes = prompt_data["attributes"]
        
        # Positive prompts (should match)
        positive_prompts = [
            f"A {attributes[obj]} {obj}" for obj in objects if obj in attributes
        ]
        
        # Negative prompts (should not match)
        negative_prompts = []
        all_colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        for obj in objects:
            if obj in attributes:
                correct_attr = attributes[obj]
                wrong_attrs = [c for c in all_colors if c != correct_attr]
                negative_prompts.extend([f"A {attr} {obj}" for attr in wrong_attrs[:2]])
        
        # Compute CLIP scores
        positive_scores = self._compute_clip_scores(image, positive_prompts)
        negative_scores = self._compute_clip_scores(image, negative_prompts)
        
        # Score is high if positive > negative
        score = float(positive_scores.mean() - negative_scores.mean())
        return max(0.0, min(1.0, (score + 1) / 2))  # Normalize to [0, 1]
    
    def verify_spatial_relation(
        self,
        image: torch.Tensor,
        prompt_data: Dict[str, any],
    ) -> float:
        """
        Verify if image correctly represents spatial relations.
        
        Args:
            image: Generated image
            prompt_data: Prompt metadata
            
        Returns:
            score: Verification score [0, 1]
        """
        relation = prompt_data["relation"]
        objects = prompt_data["objects"]
        
        # Positive: correct relation
        positive_prompt = prompt_data["text"]
        
        # Negative: opposite relation
        opposite_relations = {
            "on top of": "under",
            "under": "on top of",
            "left of": "right of",
            "right of": "left of",
            "behind": "in front of",
            "in front of": "behind",
        }
        
        opposite = opposite_relations.get(relation, "next to")
        negative_prompt = prompt_data["text"].replace(relation, opposite)
        
        # Compute scores
        positive_score = self._compute_clip_scores(image, [positive_prompt])
        negative_score = self._compute_clip_scores(image, [negative_prompt])
        
        score = float(positive_score - negative_score)
        return max(0.0, min(1.0, (score + 1) / 2))
    
    def verify_negation(
        self,
        image: torch.Tensor,
        prompt_data: Dict[str, any],
    ) -> float:
        """
        Verify if image correctly handles negation.
        
        Args:
            image: Generated image
            prompt_data: Prompt metadata
            
        Returns:
            score: Verification score [0, 1]
        """
        # Positive: prompt without negated attribute
        objects = prompt_data["objects"]
        obj = objects[0]
        
        # Negative: prompt with negated attribute
        if "negated_attribute" in prompt_data:
            negated_attr = prompt_data["negated_attribute"]
            negative_prompt = f"A {negated_attr} {obj}"
        elif "negated_objects" in prompt_data:
            negated_obj = prompt_data["negated_objects"][0]
            negative_prompt = f"A {negated_obj}"
        else:
            return 0.5
        
        # Score should be low for negated content
        negative_score = self._compute_clip_scores(image, [negative_prompt])
        
        # Lower negative score = better negation handling
        score = 1.0 - float(negative_score)
        return max(0.0, min(1.0, score))
    
    def _compute_clip_scores(
        self,
        image: torch.Tensor,
        texts: List[str],
    ) -> torch.Tensor:
        """
        Compute CLIP similarity scores.
        
        Args:
            image: Single image, shape (3, H, W)
            texts: List of text prompts
            
        Returns:
            scores: Similarity scores, shape (len(texts),)
        """
        import clip
        
        # Normalize and resize image
        image = (image + 1) / 2  # [-1, 1] -> [0, 1]
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        
        with torch.no_grad():
            # Encode image
            image_features = self.clip_model.encode_image(image.to(self.device))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Encode texts
            text_tokens = clip.tokenize(texts).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            scores = (image_features @ text_features.T).squeeze(0)
        
        return scores
    
    def evaluate_suite(
        self,
        images: torch.Tensor,
        prompts: List[Dict[str, any]],
    ) -> Dict[str, float]:
        """
        Evaluate on full compositional suite.
        
        Args:
            images: Generated images, shape (N, 3, H, W)
            prompts: List of prompt metadata
            
        Returns:
            results: Dictionary of scores by category
        """
        assert len(images) == len(prompts)
        
        category_scores = {
            "attributes": [],
            "spatial": [],
            "negation": [],
            "complex": [],
        }
        
        for image, prompt_data in zip(images, prompts):
            category = prompt_data["category"]
            
            if category == "attributes":
                score = self.verify_attribute_binding(image, prompt_data)
            elif category == "spatial":
                score = self.verify_spatial_relation(image, prompt_data)
            elif category == "negation":
                score = self.verify_negation(image, prompt_data)
            elif category == "complex":
                # Complex prompts combine multiple verifications
                attr_score = self.verify_attribute_binding(image, prompt_data)
                spatial_score = self.verify_spatial_relation(image, prompt_data)
                neg_score = self.verify_negation(image, prompt_data)
                score = (attr_score + spatial_score + neg_score) / 3
            else:
                score = 0.0
            
            category_scores[category].append(score)
        
        # Compute averages
        results = {
            f"{cat}_score": sum(scores) / len(scores) if scores else 0.0
            for cat, scores in category_scores.items()
        }
        
        results["overall_compositional_score"] = sum(results.values()) / len(results)
        
        return results
