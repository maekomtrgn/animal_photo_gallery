import os
import shutil
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import json
import urllib.request

class ImageClassifier:
    def __init__(self):
        print("Loading ResNet18 AI model for ANIMAL detection...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.to(self.device)
        self.model.eval()
        
        # Load ImageNet labels
        self.all_labels = self._get_labels()
        
        # ANIMAL-SPECIFIC CATEGORIES - Focus only on animals
        self.animal_categories = {
            'dog': self._get_animal_labels('dog'),
            'cat': self._get_animal_labels('cat'),
            'bird': self._get_animal_labels('bird'),
            'other_animal': self._get_other_animal_labels(),
            'non_animal': []  # For things that aren't animals
        }
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        print("Animal Classifier ready!")
        print(f"Target categories: {list(self.animal_categories.keys())}")

        # Simple fun-fact / description dictionary (extend as you like)
        self.fun_facts = {
            "dog": "Dogs have a sense of smell up to 100,000 times stronger than humans.",
            "cat": "Cats spend around 70% of their lives sleeping.",
            "bird": "Some birds can see ultraviolet light, which humans cannot.",
            "other_animal": "Many animals can see better at night than humans thanks to a special reflective eye layer.",
            "non_animal": "No animal detected, but that is still an interesting scene!"
        }

    def _get_labels(self):
        """Download ImageNet labels"""
        LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        try:
            with urllib.request.urlopen(LABELS_URL) as url:
                return json.loads(url.read().decode())
        except:
            # Fallback to local labels if no internet
            return [f"class_{i}" for i in range(1000)]

    def _get_animal_labels(self, animal_type):
        """Get ImageNet labels for specific animal types"""
        animal_keywords = {
            'dog': [
                'greyhound', 'golden retriever', 'Labrador retriever', 'German shepherd',
                'pug', 'chihuahua', 'beagle', 'bulldog', 'husky', 'dalmatian',
                'Great Dane', 'boxer', 'Doberman', 'rottweiler', 'schnauzer',
                'Shetland sheepdog', 'whippet', 'bloodhound', 'malamute', 'papillon',
                'Shih-Tzu', 'Afghan hound', 'basset', 'cocker spaniel', 'collie'
            ],
            'cat': [
                'tabby', 'Persian cat', 'Siamese cat', 'Egyptian cat', 'tiger cat',
                'lynx', 'leopard', 'lion', 'tiger', 'cheetah', 'snow leopard',
                'jaguar', 'cougar', 'cat'
            ],
            'bird': [
                'robin', 'junco', 'brambling', 'sparrow', 'eagle', 'vulture',
                'peacock', 'flamingo', 'ostrich', 'penguin', 'bald eagle',
                'golden eagle', 'black grouse', 'ptarmigan', 'ruffed grouse',
                'prairie chicken', 'peacock', 'quail', 'partridge', 'African grey',
                'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater',
                'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser',
                'goose', 'black swan', 'white stork', 'black stork', 'spoonbill',
                'flamingo', 'little blue heron', 'American egret', 'bittern'
            ]
        }
        return animal_keywords.get(animal_type, [])

    def _get_other_animal_labels(self):
        """Get labels for other animals (not dog/cat/bird)"""
        return [
            # Mammals
            'giant panda', 'red panda', 'raccoon', 'skunk', 'weasel', 'mink',
            'polecat', 'black-footed ferret', 'otter', 'badger', 'armadillo',
            'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon',
            'siamang', 'guenon', 'patas', 'baboon', 'macaque', 'langur',
            'colobus', 'proboscis monkey', 'marmoset', 'capuchin', 'howler monkey',
            'titi', 'spider monkey', 'squirrel monkey', 'Madagascar cat',
            'indri', 'Asian elephant', 'African elephant', 'red fox', 'kit fox',
            'Arctic fox', 'grey fox', 'tabby', 'tiger', 'cheetah', 'lion',
            'snow leopard', 'jaguar', 'cougar', 'lynx', 'leopard',
            
            # Marine animals
            'dugong', 'sea lion', 'Chihuahua', 'whale', 'killer whale',
            'bottlenose dolphin', 'seal', 'walrus',
            
            # Farm animals
            'ox', 'water buffalo', 'bison', 'ram', 'bighorn', 'ibex',
            'hartebeest', 'impala', 'gazelle', 'Arabian camel', 'llama',
            'weasel', 'mink', 'European polecat', 'black-footed ferret',
            'otter', 'skunk', 'badger', 'armadillo', 'three-toed sloth',
            'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'siamang',
            'guenon', 'patas', 'baboon', 'macaque'
        ]

    def is_animal(self, label):
        """Check if a label corresponds to an animal"""
        label_lower = label.lower()
        
        # Check all animal categories
        for category in ['dog', 'cat', 'bird', 'other_animal']:
            for keyword in self.animal_categories[category]:
                if keyword.lower() in label_lower:
                    return True
        
        # Additional animal keywords
        animal_keywords = [
            'animal', 'mammal', 'bird', 'fish', 'reptile', 'amphibian',
            'insect', 'arachnid', 'crustacean', 'mollusk', 'rodent',
            'carnivore', 'herbivore', 'omnivore', 'vertebrate', 'invertebrate'
        ]
        
        return any(keyword in label_lower for keyword in animal_keywords)

    def classify_and_organize(self, image_path):
        """
        Classify image and organize into ANIMAL categories
        Returns: (folder, detected_label, confidence)
        """
        try:
            # Open and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img)
            batch_t = torch.unsqueeze(img_t, 0)
            batch_t = batch_t.to(self.device)

            # Run inference
            with torch.no_grad():
                output = self.model(batch_t)
            
            # Get top 5 predictions
            probabilities = F.softmax(output[0], dim=0)
            top_probs, top_cat_ids = torch.topk(probabilities, 5)
            
            # Find the best animal match
            best_animal = None
            best_confidence = 0
            best_label = ""
            
            for i in range(len(top_cat_ids)):
                label = self.all_labels[top_cat_ids[i].item()]
                confidence = top_probs[i].item()
                
                # Check if this is an animal
                if self.is_animal(label):
                    if confidence > best_confidence:
                        best_animal = label
                        best_confidence = confidence
                        best_label = label
            
            # If no animal detected in top 5, check all predictions
            if best_animal is None:
                # Get all predictions with confidence > 0.01
                for idx, prob in enumerate(probabilities):
                    if prob > 0.01:  # 1% threshold
                        label = self.all_labels[idx]
                        if self.is_animal(label):
                            best_animal = label
                            best_confidence = prob.item()
                            best_label = label
                            break
            
            # Determine category
            destination_folder = 'non_animal'
            detected_label = "Non-animal object"
            
            if best_animal:
                detected_label = best_animal
                
                # Check specific categories
                for category in ['dog', 'cat', 'bird']:
                    for keyword in self.animal_categories[category]:
                        if keyword.lower() in best_animal.lower():
                            destination_folder = category
                            break
                    if destination_folder != 'non_animal':
                        break
                
                # If not dog/cat/bird but still animal
                if destination_folder == 'non_animal':
                    destination_folder = 'other_animal'
            else:
                # Not an animal at all
                best_confidence = top_probs[0].item()
                detected_label = self.all_labels[top_cat_ids[0].item()]
                print(f"  ⚠ No animal detected. Top result: {detected_label} ({best_confidence:.2%})")

            # Create gallery folder structure
            gallery_base = 'static/gallery'
            os.makedirs(gallery_base, exist_ok=True)
            
            for folder in ['dog', 'cat', 'bird', 'other_animal', 'non_animal']:
                os.makedirs(f'{gallery_base}/{folder}', exist_ok=True)

            # Move file to appropriate folder
            filename = os.path.basename(image_path)
            new_path = f'{gallery_base}/{destination_folder}/{filename}'
            shutil.move(image_path, new_path)
            
            # Save metadata
            self._save_metadata(new_path, detected_label, best_confidence, destination_folder)
            
            # Print results
            print(f"  Top 3 predictions:")
            for i in range(min(3, len(top_cat_ids))):
                label = self.all_labels[top_cat_ids[i].item()]
                prob = top_probs[i].item()
                animal_flag = "🐾" if self.is_animal(label) else "  "
                print(f"    {animal_flag} {label}: {prob:.2%}")

            return destination_folder, detected_label, best_confidence

        except Exception as e:
            print(f"Classification error: {e}")
            # Move to error folder as fallback
            os.makedirs('static/gallery/error', exist_ok=True)
            error_path = f'static/gallery/error/{os.path.basename(image_path)}'
            if os.path.exists(image_path):
                shutil.move(image_path, error_path)
            return 'error', str(e), 0.0

    def _save_metadata(self, image_path, label, confidence, category):
        """Save classification metadata as JSON"""
        import json
        from datetime import datetime
        # Choose a description based on category (fall back to stripped key)
        key = category if category in getattr(self, 'fun_facts', {}) else category.replace('_', '')
        description = getattr(self, 'fun_facts', {}).get(key, '')

        metadata = {
            'filename': os.path.basename(image_path),
            'ai_label': label,
            'confidence': f"{confidence:.2%}",
            'category': category,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'is_animal': category != 'non_animal'
        }
        
        metadata_path = image_path.replace('.jpg', '.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

# Test function
def test_classifier():
    """Test the animal classifier with sample images"""
    classifier = ImageClassifier()
    
    # Test with different types of images
    test_cases = [
        ("A dog photo", "dog"),
        ("A cat photo", "cat"),
        ("A bird photo", "bird"),
        ("A car (non-animal)", "non_animal")
    ]
    
    print("\n" + "="*60)
    print("ANIMAL CLASSIFIER TEST")
    print("="*60)
    
    for description, expected in test_cases:
        print(f"\nTesting: {description}")
        print(f"Expected: {expected}")
        # Note: You would need actual test images here
        print("(Add test images to test this functionality)")

if __name__ == "__main__":
    test_classifier()
