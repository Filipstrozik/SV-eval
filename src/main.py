

from torch.utils.data import DataLoader
from utils import *



dataset_path = '../data/voxceleb2_wav'
print('Scanning dataset...')
df = scan_directory_voxceleb2(dataset_path)


model_name = 'campplus'
print('Loading model...')
model = load_model(model_name)

max_len = 5 * 16000
audio_dataset = AudioDatasetFBank(df, max_len, model)
audio_loader = DataLoader(audio_dataset, batch_size=32, shuffle=True)
print('Evaluating model...')
embeddings = evaluate_wespeaker_fbank(model, audio_loader)
print('Saving embeddings...')
save_embeddings_to_csv(df, embeddings, f'../embeds/{model_name}_embeddings.csv')
print('Done!')

df = pd.read_csv(f'../embeds/{model_name}_embeddings.csv')
df = preprocess_embeddings_df(df)

scores, class_labels = calculate_cosine_similarity_matrix(df)
EER, EER_threshold, thresholds, FAR, FRR = calculate_eer(scores, class_labels)
print(f'EER: {EER} with threshold {EER_threshold}')

plot_frr_far(FAR, FRR, EER, EER_threshold, thresholds)
# save results to json EER, EER_threshold, model_name
results = {'EER': EER, 'EER_threshold': EER_threshold, 'model_name': model_name}