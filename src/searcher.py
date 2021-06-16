from torch import nn
import torch
from models import Synth
import faiss
from caffe2torch import SketchModel, bwmorph_thin
import torch.nn.functional as F
from utils import *
from dataset import *
import cv2


class Searcher:

    def __init__(self, dataset_name, index_name, root=None):
        self.device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        # Synthesis Model
        self.model = Synth()
        self.model = nn.DataParallel(self.model)
        weights_filepath = os.path.join(os.path.dirname(__file__), '../weights/synth_gnet.pt')
        self.model.load_state_dict(torch.load(weights_filepath, map_location=self.device))
        self.model.eval().to(self.device)
        self.f_size = 832

        # Sketch model
        cag_weight = os.path.join(os.path.dirname(__file__), 'caffe2torch/model_sketch2.npy')
        self.skt_hasher = SketchModel(cag_weight).to(self.device)

        # Load Faiss index of OpenImages test set
        self.index = faiss.read_index(os.path.join(os.path.dirname(__file__), f'../indexes/{index_name}'))

        self.custom = dataset_name == 'Custom'
        # Dataset
        if dataset_name == 'Custom':
            self.ds = CustomDataset(root)
        elif dataset_name == 'OpenImages':
            self.ds = OpenImages()
        else:
            self.ds = Unsplash()

    def extract_descriptor(self, img):
        skt = self.sketch_loader(img)
        d = self.skt_hasher.inference([skt])
        return d

    def search(self, img, k=30):
        queries = torch.zeros(1, 256, 31, 31)
        qbs = []

        sketches, qmasks, fmasks = self.separate_sketches(img)

        fqs = torch.zeros(1, self.f_size * 7 * 7).to(self.device)
        for sketch, qb, mb in zip(sketches, qmasks, fmasks):
            masks = torch.zeros(1, self.f_size, 7, 7)
            d = self.extract_descriptor(sketch)
            qbs.append(qb)
            for w in range(qb[0], qb[1]):
                for h in range(qb[2], qb[3]):
                    queries[0, :, h, w] = torch.Tensor(d)
            masks[0, :, mb[0]:mb[1], mb[2]:mb[3]] = 1

            # Synthesise visual feature
            fQ = self.model(queries.to(self.device)) * masks.to(self.device)
            fQ = fQ.flatten(1)
            fQ = F.normalize(fQ, p=2, dim=1)
            fqs[torch.abs(fQ) > fqs] = fQ[torch.abs(fQ) > fqs]
        fQ = fqs
        fQ = F.normalize(fQ, p=2, dim=1)
        fQ = fQ.detach().cpu().numpy().astype('float32')

        D, I = self.index.search(fQ, k)

        urls = []
        for idx in I[0]:
            if self.custom:
                img = self.ds.get_original(idx)
                urls.append(img)
            else:
                batch = self.ds[idx]
                urls.append(batch['url'])

        return urls, qmasks

    @staticmethod
    def sketch_loader(sk):
        """
        read sketch using PIL, centralise and make a square image
        """
        if type(sk) is np.ndarray:
            img = Image.fromarray(sk)
        else:
            with open(sk, 'rb') as f:
                img = Image.open(f).convert('L')
        img = 255 - np.array(img)  # background is black
        # img = cv2.dilate(img, st)

        # crop object
        nz = np.nonzero(img)
        ymin = max(0, nz[0].min() - 1)
        ymax = min(img.shape[0], nz[0].max() + 1)
        xmin = max(0, nz[1].min() - 1)
        xmax = min(img.shape[1], nz[1].max() + 1)
        img = img[ymin:ymax, xmin:xmax]
        # resize
        sf = 200. / max(img.shape)
        h, w = int(img.shape[0] * sf), int(img.shape[1] * sf)
        img = Image.fromarray(img).resize((w, h), Image.BILINEAR)
        img = np.array(img)
        # thinning
        img = bwmorph_thin(img > 50).astype(np.float32) * 255
        # padding to 224x224
        h, w = img.shape
        dx = (224 - w) // 2
        dy = (224 - h) // 2
        img = np.pad(img, ((dy, 224 - dy - h), (dx, 224 - dx - w)))
        # invert background color
        img = np.float32(255 - img)
        # Image.fromarray(img).convert('RGB').save('sk_test.png')
        # gray2BGR
        img = np.repeat(img[..., None], 3, axis=2)

        return img

    @staticmethod
    def separate_sketches(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

        im_h, im_w, im_c = img.shape
        trans_mask = img[:, :, 3] != 255
        img[trans_mask] = [10, 100, 130, 255]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img[img == 0] = 47
        image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, _, _ = cv2.split(image_hsv)
        peaks = np.unique(h)
        image = image_hsv
        # Now let's find the shape matching each dominant hue
        blobs = []
        qmasks = []
        fmasks = []
        for i, peak in enumerate(peaks):
            # First we create a mask selecting all the pixels of this hue
            mask = cv2.inRange(h, int(peak - 1), int(peak + 1))
            # And use it to extract the corresponding part of the original colour image
            blob = cv2.bitwise_and(image, image, mask=mask)
            hh, ww, cc = np.where(blob != 0)
            bbox = [np.min(ww), np.min(hh), np.max(ww), np.max(hh)]
            if bbox[3] - bbox[1] < 10 or bbox[2] - bbox[0] < 10:
                break
            blob = blob[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2GRAY)
            blob[blob != 0] = 255
            blob = 255 - blob

            qmask = [np.floor(bbox[1] / im_h * 31),
                     np.ceil(bbox[3] / im_h * 31),
                     np.floor(bbox[0] / im_w * 31),
                     np.ceil(bbox[2] / im_w * 31)]
            qmask = [int(x) for x in qmask]

            fmask = [int(np.round(x / 31 * 7)) for x in qmask]

            # Remove background blob
            if image[:, :, 0].size - 77500 > blob.size:
                blobs.append(blob)
                qmasks.append(qmask)
                fmasks.append(fmask)

        return blobs, qmasks, fmasks
