import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

from utils import augmentation
from utils.augmentation import Image

sizes_hierarchy = {
    'finegym': (307, [288, 15, 4]),
    'hollywood2': (17, [12, 5])
}


def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    except:
        print(f'Error with image in path {path}')
        return Image.new('RGB', (150, 150))  # zero image


class Kinetics600(data.Dataset):
    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 return_label=False,
                 return_idx=False,
                 path_dataset='',
                 path_data_info=''):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.return_label = return_label
        self.return_idx = return_idx

        self.path_dataset = path_dataset
        self.path_data_info = path_data_info

        # splits
        if mode == 'train':
            split = os.path.join(self.path_data_info, 'kinetics600/train_split.csv')
            video_info = pd.read_csv(split, header=None)
        elif (mode == 'val') or (mode == 'test'):
            split = os.path.join(self.path_data_info, 'kinetics600/train_split.csv')
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        path_drop_idx = os.path.join(self.path_data_info, f'drop_idx_kinetics_{mode}.pth')
        if os.path.isfile(path_drop_idx):
            drop_idx = torch.load(path_drop_idx)
        else:
            drop_idx = []
            print('filter out too short videos ...')
            for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
                vpath, vlen = row
                if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                    drop_idx.append(idx)
            torch.save(drop_idx, path_drop_idx)

        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode == 'val':
            self.video_info = self.video_info.sample(frac=0.3, random_state=666)
        if mode == 'test':
            self.video_info = self.video_info.sample(1000, random_state=666)
        # shuffle not necessary because use RandomSampler

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0: return [None]
        n = 1
        start_idx = np.random.choice(range(vlen - self.num_seq * self.seq_len * self.downsample), n)
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        items = self.idx_sampler(vlen, vpath)
        while items is None or items[0] is None:
            index = random.randint(0, len(self.video_info) - 1)
            vpath, vlen = self.video_info.iloc[index]
            items = self.idx_sampler(vlen, vpath)

        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1))) for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)
        if self.return_idx:
            return t_seq, 0, index
        return t_seq, 0  # placeholder for labels, need implement

    def __len__(self):
        return len(self.video_info)


class Hollywood2(data.Dataset):
    """
    Number of classes: 12
    Number of classes including parents: 17 (12 + 5)
    """

    def __init__(self,
                 mode='train',
                 transform=None,
                 seq_len=10,
                 num_seq=5,
                 downsample=3,
                 epsilon=5,
                 return_label=False,
                 hierarchical_label=False,
                 path_dataset='',
                 path_data_info=''):
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon
        self.return_label = return_label
        self.hierarchical_label = hierarchical_label
        self.path_dataset = path_dataset
        self.path_data_info = path_data_info

        if mode in ['train', 'val']:
            split = os.path.join(self.path_data_info, 'hollywood2/train_split.csv')
            video_info = pd.read_csv(split, header=None)
        elif mode == 'test':
            split = os.path.join(self.path_data_info, 'hollywood2/test_split.csv')
            video_info = pd.read_csv(split, header=None)
        else:
            raise ValueError('wrong mode')

        path_drop_idx = os.path.join(self.path_data_info, f'hollywood2/drop_idx_hollywood_{mode}.pth')
        if os.path.isfile(path_drop_idx):
            drop_idx = torch.load(path_drop_idx)
        else:
            drop_idx = []
            print('filter out too short videos ...')
            for idx, row in tqdm(video_info.iterrows(), total=len(video_info)):
                vpath, vlen = row
                if vlen - self.num_seq * self.seq_len * self.downsample <= 0:
                    drop_idx.append(idx)
            torch.save(drop_idx, path_drop_idx)

        self.video_info = video_info.drop(drop_idx, axis=0)

        if mode in ['train', 'val']:
            train_info, val_info = train_test_split(self.video_info, test_size=0.1, random_state=666)
            self.video_info = train_info if mode == "train" else val_info
        # shuffle not necessary because use RandomSampler

        # Read action and index
        self.dict_labels = {}
        self.dict_labels_hier = {}
        label_path = os.path.join(self.path_dataset, 'hollywood2/class_Ind')
        with open(os.path.join(label_path, 'class_Ind.txt'), 'r') as f:
            for line in f:
                action, label = line.split()
                self.dict_labels[action] = label

        with open(os.path.join(label_path, 'class_Ind_Hier.txt'), 'r') as f:
            for line in f:
                action, label, _ = line.split()
                self.dict_labels_hier[action] = label

        self.child_nodes = defaultdict(list)
        self.parent_nodes = defaultdict(list)
        with open(os.path.join(label_path, 'class_Relation.txt'), 'r') as f:
            for line in f:
                parent, child = line.split()
                self.child_nodes[parent].append(child)
                self.parent_nodes[child].append(parent)

        self.hierarchy = defaultdict(list)
        with open(os.path.join(label_path, 'class_Ind_Hier.txt'), 'r') as f:
            for line in f:
                action, label, level = line.split()
                self.hierarchy[level].append(action)
        # Get labels
        self.labels = {}
        with open(os.path.join(self.path_data_info, 'hollywood2/hollywood2_videos.txt'), 'r') as f:
            for line in f:
                key, label, *_ = line.split()
                if '-' in label:  # scene, not action
                    continue
                key = key.split('/')[-1].split('.')[0]
                self.labels[key] = label

    def idx_sampler(self, vlen, vpath):
        '''sample index from a video'''
        if vlen - self.num_seq * self.seq_len * self.downsample <= 0: return [None]
        n = 1
        # image index starts from 1
        if self.mode == 'train':
            start_idx = np.random.choice(range(vlen - self.num_seq * self.seq_len * self.downsample + 1), n)
        else:
            start_idx = (vlen - self.num_seq * self.seq_len * self.downsample + 1) // 2
        seq_idx = np.expand_dims(np.arange(self.num_seq), -1) * self.downsample * self.seq_len + start_idx
        seq_idx_block = seq_idx + np.expand_dims(np.arange(self.seq_len), 0) * self.downsample
        return [seq_idx_block, vpath]

    def __getitem__(self, index):
        vpath, vlen = self.video_info.iloc[index]
        action = self.labels[vpath.split('/')[-1]]
        label = torch.LongTensor([int(self.dict_labels[action])])

        items = self.idx_sampler(vlen, vpath)
        if items is None: print(vpath)

        idx_block, vpath = items
        assert idx_block.shape == (self.num_seq, self.seq_len)
        idx_block = idx_block.reshape(self.num_seq * self.seq_len)

        seq = [pil_loader(os.path.join(vpath, 'image_%05d.jpg' % (i + 1))) for i in idx_block]
        t_seq = self.transform(seq)  # apply same transform
        (C, H, W) = t_seq[0].size()
        t_seq = torch.stack(t_seq, 0)
        t_seq = t_seq.view(self.num_seq, self.seq_len, C, H, W).transpose(1, 2)
        if self.return_label and not self.hierarchical_label:
            return t_seq, label
        if self.return_label and self.hierarchical_label:
            labels = []
            actions = []
            while action != 'Root':
                labels.append(torch.LongTensor([int(self.dict_labels_hier[action])]))
                actions.append(action)
                action = self.parent_nodes[action][0]
            labels = torch.cat(labels)
        else:
            labels = torch.tensor(-1)
        return t_seq, labels, index

    def __len__(self):
        return len(self.video_info)


class FineGym(data.Dataset):
    """
    If we select gym288, the number of classes to predict is:
    - 288 in the subaction level
    - 4 in the action level
    - 307 in the hierarchical level (288 + 15 + 4)
    """

    def __init__(self,
                 mode='train',
                 path_dataset='',
                 transform=None,
                 seq_len=10,  # given duration distribution, we should aim for ~1.5 seconds (around 7-8 frames at 5 fps)
                 num_seq=5,
                 epsilon=5,
                 return_label=False,
                 gym288=True,
                 fps=5,
                 hierarchical_label=False,
                 action_level_gt=False,
                 return_idx=False,
                 path_data_info=''):
        self.path_dataset = path_dataset
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.epsilon = epsilon
        self.return_label = return_label
        self.fps = fps
        self.hierarchical_label = hierarchical_label,
        self.action_level_gt = action_level_gt
        self.return_idx = return_idx
        self.path_data_info = path_data_info

        if mode in ['train', 'val']:
            path_labels = 'gym288_train_element_v1.1.txt' if gym288 else 'gym99_train_element_v1.1.txt'
        elif mode == 'test':
            path_labels = 'gym288_val_element.txt' if gym288 else 'gym99_val_element.txt'
        else:
            raise ValueError('wrong mode')
        path_labels = 'annotations/' + path_labels

        with open(os.path.join(self.path_data_info, 'finegym/annotations/finegym_annotation_info_v1.1.json'), 'r') as f:
            self.annotations = json.load(f)

        # Prepare superclasses
        # First, load information about the available superclasses
        self.parent_classes = {}
        with open(os.path.join(self.path_data_info, 'finegym/categories/set_categories.txt'), 'r') as f:
            class_number = 288 if gym288 else 99
            set_grand_classes = set()
            for line in f:
                _, idx_class, name_class, _ = re.split(': |; |\\n', line)
                # If there's only one element in the grand class, we still treat it as separate classes
                grand_class = idx_class[0]
                self.parent_classes[idx_class] = (class_number, name_class, grand_class)
                set_grand_classes.add(grand_class)
                class_number += 1
        self.grand_classes = {name: class_number + i for i, name in enumerate(sorted(set_grand_classes))}

        self.super_classes = {}
        path_categories = 'gym288_categories.txt' if gym288 else 'gym99_categories.txt'
        with open(os.path.join(self.path_data_info, 'finegym/categories', path_categories), 'r') as f:
            for line in f:
                line_split = re.split(': |; ', re.sub(' +', ' ', line))
                subclass = int(line_split[1])
                superclass_parent_idx = line_split[3]
                superclass_grand_idx = superclass_parent_idx[0]
                self.super_classes[subclass] = (superclass_parent_idx, superclass_grand_idx)

        clips = []
        for root, dirs, files in os.walk(os.path.join(path_dataset, 'event_videos')):
            # We look at this folder instead of self.annotations because not all videos are downloaded for now
            for file in files:
                clips.append(file.replace('.mp4', ''))

        self.subclipidx2label = {}
        clips_in_labels = set()  # Some clips in "clips" may belong to another split or not even be in the *element.txt
        with open(os.path.join(self.path_data_info, 'finegym', path_labels), 'r') as f:
            for line in f:
                data_split = line.replace('\n', '').split()
                subclip_name = data_split[0]
                self.subclipidx2label[subclip_name] = int(data_split[1])
                clip_name = subclip_name[:27]
                clips_in_labels.add(clip_name)

        self.clips = {}  # Actual clips used in the dataset, with its actions
        for clip in clips:
            if clip.startswith('.'):
                continue
            if self.return_label and clip not in clips_in_labels:  # For gym288, this filters out almost 2/3 of the data
                continue
            assert len(clip) == 27  # youtube ID is 11, event ID is 15, and the separation
            video_id = clip[:11]
            event_id = clip[12:]
            segments = self.annotations[video_id][event_id]['segments']

            if segments is not None and (
                    # This is filtering out several short videos, approx 1/3 of the remaining videos for self.num_seq=6
                    len(segments) >= self.num_seq if self.return_label else
                    np.array([s['stages'] for s in segments.values()]).sum() >= self.num_seq):
                if np.array([s['stages'] > 1 for s in segments.values()]).any() and not self.return_label:
                    segments = {k1 + f'_{i}': {'timestamps': v1['timestamps'][i]} for k1, v1 in segments.items()
                                for i in range(v1['stages'])}
                self.clips[clip] = segments

        if mode in ['train', 'val']:
            labels_val = random.Random(500).sample(range(len(self.clips)), int(0.2 * len(self.clips)))
            if mode == 'val':  # take only 20% of the labels of the "train" split
                self.clips = {k: v for i, (k, v) in enumerate(self.clips.items()) if i in labels_val}
            else:  # mode == 'train':  # take the other 80% of the "train" split
                labels_train = list(set(range(len(self.clips))) - set(labels_val))
                self.clips = {k: v for i, (k, v) in enumerate(self.clips.items()) if i in labels_train}

        self.idx2clipidx = {i: clipidx for i, clipidx in enumerate(self.clips.keys())}

    def read_video(self, clipidx, segments):
        # Sample self.num_seq consecutive actions from this segment
        if self.mode == 'train':
            start = random.randint(0, len(segments) - self.num_seq)
        else:
            start = (len(segments) - self.num_seq) // 2
        actions = list(segments.keys())
        total_clip = []
        labels = []
        for i in range(start, start + self.num_seq):
            subclipidx = clipidx + '_' + actions[i]
            subfolder = 'action_videos' if len(actions[i]) == 11 else 'stage_videos'
            path_clip = os.path.join(self.path_dataset, subfolder, f'{subclipidx}.mp4')
            if os.path.isfile(path_clip):
                video, audio, info = torchvision.io.read_video(path_clip, start_pts=0, end_pts=None, pts_unit='sec')
                video = video.float()
                # Adapt to fps
                step = int(np.round(info['video_fps'] / self.fps))
                video_resampled = video[range(0, video.shape[0], step)]
                # If the video is too long, trim (random position or middle, depending on mode)
                if self.mode == 'train':
                    start_subclip = random.randint(0, np.maximum(0, len(video_resampled) - self.seq_len))
                else:
                    start_subclip = np.maximum(0, len(video_resampled) - self.seq_len) // 2
                video_trimmed = video_resampled[start_subclip:start_subclip + self.seq_len]
                # [C T H W] is the format for the torchvision._transforms_video
                video_resampled = video_trimmed.permute(3, 0, 1, 2)
                # We transform at the subclip level. No need to have transformation consistency between clips
                video_transformed = self.transform(video_resampled)  # apply same transform
                # Zero-pad short clips
                padding = [0, ] * 8
                padding[5] = self.seq_len - video_transformed.shape[1]
                video_padded = torch.nn.functional.pad(video_transformed, pad=padding, mode="constant", value=0)
            else:
                # print(f'{path_clip} is not a valid file')
                video_padded = torch.zeros((3, self.seq_len, 128, 128))
            total_clip.append(video_padded)

            if subclipidx not in self.subclipidx2label:
                # This happens when the specific case when the action is not part of the action classes (for example
                # when it is very specific and we are working with gym99). In this case we still load the action because
                # if we skip it the temporal prediction does not make sense.
                labels.append(-1 if not self.hierarchical_label else torch.tensor([-1] * 3))
            else:
                if self.hierarchical_label or self.action_level_gt:
                    label_specific = self.subclipidx2label[subclipidx]
                    p_idx, g_idx = self.super_classes[label_specific]
                    labels.append(torch.tensor([label_specific, self.parent_classes[p_idx][0],
                                                self.grand_classes[g_idx]]))
                else:
                    labels.append(self.subclipidx2label[subclipidx])

        total_clip = torch.stack(total_clip)
        labels = torch.stack(labels)

        if self.action_level_gt:
            # All the subclips should have the same action grandparent, unless there's some "-1"
            labels_to_consider = labels[:, -1][labels[:, -1] != -1]
            if len(labels_to_consider) > 0:
                assert torch.all(labels_to_consider == labels_to_consider[0]), 'What is going on?'
                labels = labels_to_consider[0] - 288 - 15
            else:
                labels = torch.tensor(-1)

        return total_clip, labels

    def __getitem__(self, index):
        clipidx = self.idx2clipidx[index]
        segments = self.clips[clipidx]
        video, labels = self.read_video(clipidx, segments)
        if not self.return_label:
            labels = torch.tensor(-1)
        if self.return_idx:
            return video, labels, index
        return video, labels, index

    def __len__(self):
        return len(self.clips)


class MovieNet(data.Dataset):
    def __init__(self, mode='train', transform=None, num_seq=5, path_dataset='', path_data_info=''):
        self.path_dataset = path_dataset
        self.mode = mode
        self.transform = transform
        self.num_seq = num_seq
        self.path_data_info = path_data_info

        path_save = os.path.join(self.path_data_info, f'movienet_{mode}.pth')

        if os.path.isfile(path_save):
            self.clips, self.subclip_seqs = torch.load(path_save)
        else:
            self.clips = defaultdict(lambda: defaultdict(list))
            for root, dirs, files in tqdm(os.walk(os.path.join(self.path_dataset, '240P'))):
                for file in files:
                    if file.endswith('.jpg'):
                        video_num = root.split('/')[-1]
                        _, clip_num, _, frame_num = file.replace('.jpg', '').split('_')
                        self.clips[video_num][int(clip_num)].append(int(frame_num))

            randomized_indices = list(range(len(self.clips)))
            random.Random(500).shuffle(randomized_indices)
            low, high = {'train': [0, 0.8], 'val': [0.8, 0.9], 'test': [0.9, 1]}[self.mode]
            labels_mode = randomized_indices[int(low * len(self.clips)):int(high * len(self.clips))]
            self.clips = {k: v for i, (k, v) in enumerate(self.clips.items()) if i in labels_mode}

            # split clips into subclip sequences of num_seq elements
            self.subclip_seqs = []
            for k, v in self.clips.items():
                all_clips = np.sort(list(v.keys()))
                all_clips = all_clips[:num_seq * (len(all_clips) // num_seq)].reshape(len(all_clips) // num_seq,
                                                                                      num_seq)
                for i in range(all_clips.shape[0]):
                    self.subclip_seqs.append((all_clips[i], k, i))

            torch.save((self.clips, self.subclip_seqs), path_save)

    def read_video(self, subclip_idxs, video_idx):

        path_clip = os.path.join(self.path_dataset, '240P', video_idx)
        total_clip = []
        for subclip_idx in subclip_idxs:
            frame_list = np.sort(self.clips[video_idx][subclip_idx])
            assert len(frame_list) == 3
            for frame in frame_list:
                img = Image.open(os.path.join(path_clip, f'shot_{subclip_idx:04d}_img_{frame}.jpg'))
                total_clip.append(img)

        total_clip = torch.stack(self.transform(total_clip))  # apply same transform
        total_clip = total_clip.view([len(subclip_idxs), 3] + list(total_clip[0].shape[-3:]))

        return total_clip

    def __getitem__(self, index):
        subclip_idxs, video_idx, seq_idx = self.subclip_seqs[index]
        video = self.read_video(subclip_idxs, video_idx)
        label = torch.tensor(-1)
        return video, label, index

    def __len__(self):
        return len(self.subclip_seqs)


def get_data(args, mode='train', return_label=False, hierarchical_label=False, action_level_gt=False, \
             num_workers=0, path_dataset='', path_data_info=''):
    if hierarchical_label and args.dataset not in ['finegym', 'hollywood2']:
        raise Exception('Hierarchical information is only implemented in finegym and hollywood2 datasets')
    if return_label and not action_level_gt and args.dataset != 'finegym':
        raise Exception('subaction only subactions available in finegym dataset')

    if mode == 'train':
        transform = transforms.Compose([
            augmentation.RandomSizedCrop(size=args.img_dim, consistent=True, p=1.0),
            augmentation.RandomHorizontalFlip(consistent=True),
            augmentation.RandomGray(consistent=False, p=0.5),
            augmentation.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=1.0),
            augmentation.ToTensor(),
            augmentation.Normalize()
        ])
    else:
        transform = transforms.Compose([
            augmentation.CenterCrop(size=args.img_dim, consistent=True),
            augmentation.ToTensor(),
            augmentation.Normalize()
        ])

    if args.dataset == 'kinetics':
        dataset = Kinetics600(mode=mode,
                              transform=transform,
                              seq_len=args.seq_len,
                              num_seq=args.num_seq,
                              downsample=5,
                              return_label=return_label,
                              return_idx=False,
                              path_dataset=path_dataset,
                              path_data_info=path_data_info)
    elif args.dataset == 'hollywood2':
        if return_label:
            assert action_level_gt, 'hollywood2 does not have subaction labels'
        dataset = Hollywood2(mode=mode,
                             transform=transform,
                             seq_len=args.seq_len,
                             num_seq=args.num_seq,
                             downsample=args.ds,
                             return_label=return_label,
                             hierarchical_label=hierarchical_label,
                             path_dataset=path_dataset,
                             path_data_info=path_data_info)
    elif args.dataset == 'finegym':
        if hierarchical_label:
            assert not action_level_gt, 'finegym does not have hierarchical information at the action level'
        dataset = FineGym(mode=mode,
                          transform=transform,
                          seq_len=args.seq_len,
                          num_seq=args.num_seq,
                          fps=int(25 / args.ds),  # approx
                          return_label=return_label,
                          hierarchical_label=hierarchical_label,
                          action_level_gt=action_level_gt,
                          path_dataset=path_dataset,
                          return_idx=False,
                          path_data_info=path_data_info)
    elif args.dataset == 'movienet':
        assert not return_label, 'Not yet implemented (actions not available online)'
        assert args.seq_len == 3, 'We only have 3 frames per subclip/scene, but always 3'
        dataset = MovieNet(mode=mode, transform=transform, num_seq=args.num_seq, path_dataset=path_dataset,
                           path_data_info=path_data_info)
    else:
        raise ValueError('dataset not supported')

    sampler = data.RandomSampler(dataset) if mode == 'train' else data.SequentialSampler(dataset)

    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=sampler,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=(mode != 'test')  # test always same examples independently of batch size
                                  )
    return data_loader
