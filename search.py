import faiss
import pandas as pd

import classifier
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import lime.lime_image as lime_image
import h5py
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def retrieve(database, query, k):
    # distance = np.sqrt(np.sum(np.square(database['features'] - query), 1))
    # order = np.argsort(distance)
    distance, order = database.search(query.reshape(1, -1), k)
    return distance.flatten(), order.flatten()


def reconstruct(model, image, attributes, alpha=0.5):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image.permute(1, 2, 0).numpy().astype('double'), model.predict_np,
                                             top_labels=40)
    # print(explanation.top_labels)
    image_parts = []
    for i, attr in enumerate(attributes):
        explanation_img, explanation_mask = explanation.get_image_and_mask(i, positive_only=True, hide_rest=True)
        if attr == 1:
            image_parts.append(explanation_mask[..., None] * explanation_img)
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    plt.imshow(sum(image_parts) / len(image_parts))
    plt.axis('off')
    plt.show()
    image_r = alpha * image.permute(1, 2, 0) + (1 - alpha) * sum(image_parts) / len(image_parts)
    plt.imshow(image_r)
    plt.axis('off')
    plt.show()
    return image_r, model(image_r.permute(2, 0, 1).unsqueeze(0).float(), return_features=True).numpy().flatten()


def main():
    attr_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                  'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
                  'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
                  'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                  'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                  'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    model = classifier.Classifier()
    model.load_state_dict(torch.load('model.torch'))
    model.eval()
    # with h5py.File('features_database.h5', 'r') as file:
    #     database = {'attributes': file['attributes'][...], 'features': file['features'][...]}
    database = faiss.read_index('feature_index.faiss')
    dataset = datasets.CelebA('', split='train', transform=transforms.Compose([transforms.Resize(64),
                                                                               transforms.ToTensor()]))
    test_dataset = datasets.CelebA('', split='test', transform=transforms.ToTensor())
    # class_size = torch.zeros(40)
    # for img, attr in dataset:
    #     class_size += attr
    # print(class_size / class_size.sum())
    # return
    # for i in range(30, 10000):
    #     attr = np.array(attr_names)[test_dataset[i][1].bool()]
    #     if 'Eyeglasses' in attr:
    #         print(i, attr)
    #         plt.imshow(test_dataset[i][0].permute(1, 2, 0))
    #         plt.title(i)
    #         plt.show()
    # return
    query_attr = torch.zeros(40)
    query_attr[attr_names.index('Smiling')] = 1
    # query_attr[attr_names.index('Blond_Hair')] = 1
    # query_attr[attr_names.index('Male')] = 1
    # query_attr = query_image_attr
    # reconstruct(model, test_dataset[8][0], query_attr)
    # return
    # plt.figure(figsize=(14, 14))
    # plt.title(f'Query: {", ".join(np.array(attr_names)[query_attr.bool()].tolist())}')
    # with torch.no_grad():
    #     query_image, query_image_attr = test_dataset[0]
    #     reconstructed, _ = reconstruct(model, query_image, query_attr)
    data = []
    for t in range(40):
        query_attr = torch.zeros(40)
        query_attr[t] = 1
        tp_all, fp_all, tn_all, fn_all, subset_accuracy, mean_ap = 0, 0, 0, 0, 0, 0
        n, k = 10, 10
        plt.figure(figsize=(20, 20))
        plt.suptitle(f'Query: {attr_names[t]}')
        for i in range(n):
            with torch.no_grad():
                query_image, query_image_attr = test_dataset[i]
                reconstructed, query = reconstruct(model, query_image, query_attr)
                matches, order = retrieve(database, query, k)
            plt.subplot(n, k + 2, (k + 2) * i + 1)
            plt.imshow(query_image.permute(1, 2, 0))
            plt.title('Original')
            plt.axis('off')
            plt.subplot(n, k + 2, (k + 2) * i + 2)
            plt.imshow(reconstructed)
            plt.title('Reconstructed')
            plt.axis('off')
            for j in range(k):
                # print(order[4 * i + j])
                plt.subplot(n, k + 2, (k + 2) * i + j + 3)
                plt.imshow(dataset[order[j]][0].permute(1, 2, 0))
                rel = query_attr.bool()
                attr = dataset[order[j]][1][rel]
                true_attr = query_image_attr[rel]
                tp = attr.eq(1).mul(true_attr.eq(1)).float().sum().item() / true_attr.size(0)
                fp = attr.eq(1).mul(true_attr.eq(0)).float().sum().item() / true_attr.size(0)
                tn = attr.eq(0).mul(true_attr.eq(0)).float().sum().item() / true_attr.size(0)
                fn = attr.eq(0).mul(true_attr.eq(1)).float().sum().item() / true_attr.size(0)
                tp_all += tp / (n * k)
                fp_all += fp / (n * k)
                tn_all += tn / (n * k)
                fn_all += fn / (n * k)
                subset_accuracy += torch.equal(attr, true_attr) / (n * k)
                if tp + fp != 0:
                    mean_ap += torch.equal(attr, true_attr) * (tp / (tp + fp)) / (n * k)
                plt.title(f'Result {j + 1}')
                plt.axis('off')
        accuracy = (tp_all + tn_all) / (tp_all + tn_all + fp_all + fn_all)
        if tp_all + fp_all == 0:
            precision = np.nan
        else:
            precision = tp_all / (tp_all + fp_all)
        if tp_all + fn_all == 0:
            recall = np.nan
        else:
            recall = tp_all / (tp_all + fn_all)
        if tp_all + fp_all + fn_all == 0:
            f1 = np.nan
        else:
            f1 = 2 * tp_all / (2 * tp_all + fp_all + fn_all)
        data.append([attr_names[t], precision, recall, f1, accuracy, subset_accuracy, mean_ap])
        print(tp_all, fp_all, tn_all, fn_all, accuracy, precision, recall, f1, subset_accuracy, mean_ap)
        plt.subplots_adjust()
        plt.savefig(f'fig/{attr_names[t]}.png')
        plt.close()
        # plt.show()
    print(data)
    df = pd.DataFrame(data=data, columns=['Attribute', 'Precision', 'Recall', 'F1', 'Accuracy', 'Subset Accuracy', 'mAP'])
    print(df)
    df.to_csv('results.csv')


if __name__ == '__main__':
    main()
