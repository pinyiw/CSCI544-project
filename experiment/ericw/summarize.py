#%%
from pathlib import Path
import json
import pandas as pd
from dotenv import load_dotenv
import openai
import os
import json
from pathlib import Path
from dataclasses import asdict

from dataset.ACOS.utils import (
    ACOSQuad, ACOSReview, ACOSPrediction,
    load_acos_reviews, openai_pred_acos, evaluate_acos_preds,
    laptop_acos_train_file, laptop_acos_dev_file,
    restaurant_acos_train_file, restaurant_acos_dev_file,
)

load_dotenv()
openai.organization = os.getenv('OPENAI_ORG_ID')
openai.api_key = os.getenv('OPENAI_KEY')

openai_model = "text-davinci-003"

output_dir = Path('outputs')
acos_preds_file = output_dir / 'yelp-acos-_ab50qdWOk0DdB6XOrBitw-200.jsonl'
aspect_group_output_file = output_dir / f'yelp-acos-_ab50qdWOk0DdB6XOrBitw-200-aspect-groups.json'
category_group_output_file = output_dir / f'yelp-acos-_ab50qdWOk0DdB6XOrBitw-200-category-groups.json'

with open(acos_preds_file, 'r') as f:
    acos_preds: list[ACOSPrediction] = [ACOSPrediction.from_dict(json.loads(line)) for line in f]

aspect_groups: dict[str, list[ACOSPrediction]] = {
    d['group_name']: [ACOSPrediction.from_dict(pred) for pred in d['group']]
	for d in json.load(open(aspect_group_output_file))
}

category_groups: dict[str, list[ACOSPrediction]] = {
    d['group_name']: [ACOSPrediction.from_dict(pred) for pred in d['group']]
	for d in json.load(open(category_group_output_file))
}

print(f'Loaded {len(acos_preds)} ACOS predictions')
print(f'Aspects: {aspect_groups.keys()}')
for aspect, preds in aspect_groups.items():
	print(f'  {aspect}: {len(preds)}')

print()
print(f'Categories: {category_groups.keys()}')
for category, preds in category_groups.items():
	print(f'  {category}: {len(preds)}')


aspects = list(aspect_groups.keys())
categories = list(category_groups.keys())
# %%
num_review = 10


aspect_summaries = {}
total_tokens_used = 0
for group_name, acos_preds in aspect_groups.items():
	prompt = f'Summarize the following restaurant reviews with at most 5 sentences. The summarization should only be about "{group_name}":\n\n'

	for acos_pred in acos_preds[:num_review]:
		prompt += f'"{acos_pred.review}"\n\n'

	resp = openai.Completion.create(
		model = openai_model,
		prompt = prompt,
		temperature = 0,
		max_tokens = 1000,
	)
	total_tokens_used += resp['usage']['total_tokens']
	summary = resp['choices'][0]['text'].strip()
	print(group_name)
	print(summary)
	print()
	aspect_summaries[group_name] = summary

print()
print(f"Total tokens used: {total_tokens_used:,d}")
print(f"Credit used: {total_tokens_used / 1000 * 0.02:.2f}")

# %%
category_summaries = {}
total_tokens_used = 0
for group_name, acos_preds in category_groups.items():
	prompt = f'Summarize the following restaurant reviews with at most 5 sentences. The summarization should only be about "{group_name}":\n\n'

	for acos_pred in acos_preds[:num_review]:
		prompt += f'"{acos_pred.review}"\n\n'

	resp = openai.Completion.create(
		model = openai_model,
		prompt = prompt,
		temperature = 0,
		max_tokens = 1000,
	)
	total_tokens_used += resp['usage']['total_tokens']
	summary = resp['choices'][0]['text'].strip()
	print(group_name)
	print(summary)
	print()
	category_summaries[group_name] = summary

print()
print(f"Total tokens used: {total_tokens_used:,d}")
print(f"Credit used: {total_tokens_used / 1000 * 0.02:.2f}")

# %%
aspect_summaries_output_file = output_dir / f'yelp-acos-_ab50qdWOk0DdB6XOrBitw-200-aspect-summaries.json'
category_summaries_output_file = output_dir / f'yelp-acos-_ab50qdWOk0DdB6XOrBitw-200-category-summaries.json'
# json.dump(aspect_summaries, open(aspect_summaries_output_file, 'w'), indent=2)
# json.dump(category_summaries, open(category_summaries_output_file, 'w'), indent=2)

#%%
'''
oysters
Acme is a popular restaurant in New Orleans known for its amazing oysters. Customers can enjoy fresh and chargrilled oysters, as well as a sampler plate. The wait can be long, but the oysters are worth it. Customers also rave about the Master Shucker, Hollywood, and the bread and butter that comes with the oysters. The restaurant also offers a 15 Dozen Club and a variety of other seafood dishes.

service
The reviews indicate that the service at the restaurant is generally good, with friendly staff and attentive waiters. Customers have also noted that the wait time can be long, but that the food is worth the wait.

chargrilled oysters
Acme Oyster House in New Orleans is renowned for its chargrilled oysters, which are cooked in butter, garlic, and cheese and served with fresh bread. Customers have praised the oysters for their big, creamy, and flavourful taste, even for those who don't usually like seafood. The atmosphere is casual and the service is good, making it a great spot for lunch. The other dishes, such as the seafood étouffée, crawfish, and poboys, are also highly recommended.

food
Acme Oyster House is a popular restaurant in New Orleans known for its fresh and flavorful oysters. Customers have praised the food, with favorites including charbroiled oysters, boo fries, and seafood platters. The atmosphere is casual and the service is friendly. Despite some reports of long wait times and salty food, overall reviews are positive.

Acme
Acme is a popular restaurant in New Orleans that is known for its chargrilled oysters. Many reviews praise the oysters and the po-boy sandwiches, but some have noted that the wait times can be long and the drinks are weak. The service is also spotty and the prices can be high.

place
This place is a popular restaurant in New Orleans, offering delicious chargrilled oysters, crawfish étouffée, gumbo, and po'boys. It is a must-visit for anyone visiting the city, and is worth the wait, as it is often busy. The staff are friendly and the food is delicious.

gumbo
Acme is a popular restaurant in New Orleans known for its chargrilled oysters, but their gumbo is also highly praised. Customers have described it as "delicious", "tasty", and "the most delicious one I had in NOLA". Other dishes such as the po-boy, fried crawfish, and boo fries are also recommended. The atmosphere is lively and the service is friendly.




QUALITY
The restaurant reviews indicate that the quality of the food is excellent, with fresh oysters, chargrilled oysters, and fried seafood all being highly praised. The service is also great, with friendly staff and reasonable prices. There is usually a line to get in, but it is worth the wait.

GENERAL
Acme Oyster House in New Orleans is known for its amazing oysters and bloody marys. The wait is not too long and the service is great. Prices are comparable to other restaurants in the French Quarter. There is usually a line to get in, but it is worth the wait. Reviews are generally positive, with customers praising the oysters and service.

STYLE_OPTIONS
At Acme, a New Orleans institution, customers can enjoy chargrilled oysters, 1/2 and 1/2 po'boys, fried crawfish tails, seafood gumbo, and beer on tap. The chargrilled oysters are a must-try and the gumbo is said to be delicious. Prices are comparable to other restaurants in the French Quarter and the wait can be long, but it is worth it.

SERVICE
The restaurant offers great service, with a quick wait time and attentive staff. The waitress was friendly and made the experience enjoyable. The Master Shucker was focused on his work but attentive enough to not keep customers waiting. The wait staff also provided a free chargrilled oyster for a birthday celebration.

PRICES
Prices at Acme Oyster House are comparable to other restaurants in the French Quarter, with a dozen chargrilled oysters costing around $15.50 and a 1/2 shrimp po-boy with a cup of seafood gumbo costing around $57.29. However, some reviewers have found the prices to be overpriced and have found better value elsewhere.


'''
