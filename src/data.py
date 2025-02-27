from datasets import load_dataset
import re

# Load the dataset
dataset = load_dataset("allenai/WildChat-1M", split="train").select(range(100))

# Safety categories from the paper (simplified)
SAFETY_CATEGORIES = {
    'illicit_behavior': r'\b(illegal|untraceable|shady|law enforcement|cops|hide)\b',
    'self_harm': r'\b(suicide|self-harm|kill myself|not continuing life)\b',
    'harassment': r'\b(hate|harass|threat|stalk|bully)\b',
    'sexual_content': r'\b(porn|sexual|nsfw|adult site)\b',
    'regulated_advice': r'\b(medical|legal|financial advice)\b'
}

def categorize_prompt(text):
    text = text.lower()
    for category, pattern in SAFETY_CATEGORIES.items():
        if re.search(pattern, text):
            return category
    return 'other'

# Modified categorization using dataset's original safety labels
def categorize_sample(sample):
    # Map dataset boolean categories to our policy categories
    category_mapping = {
        # Harassment
        'harassment': 'harassment',
        'harassment/threatening': 'harassment_threatening',
        
        # Hate
        'hate': 'hate',
        'hate/threatening': 'hate_threatening',
        
        # Self-harm
        'self-harm': 'self_harm',
        'self-harm/instructions': 'self_harm_instructions',
        'self-harm/intent': 'self_harm_intent',
        
        # Sexual
        'sexual': 'sexual',
        'sexual/minors': 'sexual_minors',
        
        # Violence
        'violence': 'violence',
        'violence/graphic': 'violence_graphic'
    }

    mod_categories = sample.get('moderation', {}).get('categories', {})
    
    # Check high-priority categories first
    if mod_categories.get('sexual/minors', False):
        return 'sexual_minors'
    
    if mod_categories.get('self-harm/intent', False):
        return 'self_harm_intent'
    
    # Check other categories
    detected_categories = [
        category_mapping[key]
        for key, value in mod_categories.items()
        if value is True and key in category_mapping
    ]
    
    if detected_categories:
        return detected_categories[0]  # Return first detected category
    
    return 'safe' if not sample.get('flagged', False) else 'other_unsafe'

# Extract training data with categories
training_data = []
for sample in dataset:
    # Get user prompt (first user message)
    user_prompt = next(msg['content'] for msg in sample['conversation'] if msg['role'] == 'user')
    
    # Categorize using dataset's original safety labels
    category = categorize_sample(sample)
    
    # Clean conversation data
    cleaned_conversation = [
        {'role': msg['role'], 'content': msg['content']}
        for msg in sample['conversation']
    ]
    
    training_data.append({
        'prompt': user_prompt,
        'category': category,
        'conversation': cleaned_conversation
    })

# Example structure
print("\nExample extracted entry:")
example = training_data[7]
print("Prompt:")
print(f'"{example["prompt"]}"\n')
print("Category:")
print(f'{example["category"]}\n')
print("Conversation:")
for msg in example['conversation']:
    print(f'{msg["role"].upper()}: {msg["content"]}')
    print('-' * 50)
