Convert caffe model to pytorch using mmdnn library
==================================================
## setup environment
```
docker run -it mmdnn/mmdnn:cpu.small bash
```

## convert image model
```
mkdir IR cag_caffe pytorch
```
Assume your caffe model + prototxt are in cag_caffe folder. IR is the intermediate folder, pytorch is the destination folder.

First, convert caffe model to IR (intermediate representation)
```
mmtoir -f caffe -n cag_caffe/deploy_images_net1_InceptionV1_InceptionV1_halfshare_inception4e_ld256_triplet_sketchy.prototxt -w cag_caffe/triplet1_InceptionV1_InceptionV1_halfshare_inception4e_ld256_triplet_sketchy_iter_31200.caffemodel -o IR/image
```
this will create image.pb, image.json, image.npy under IR.

Second, convert IR to pytorch:
```
mmtocode -f pytorch -n IR/image.pb -w IR/image.npy -d pytorch/model_image.py -dw pytorch/model_image.npy
```
this will create model definition (model_image.py) and model weight (model_image.npy)

Third, edit load weight method in model definition to work with numpy > 1.6, changing:
```
weights_dict = np.load(weight_file).item()
```

to:
```
weights_dict = np.load(weight_file, allow_pickle=True).item()
```