### When generating synthetic images from a trained models:

#### Put models in directory:
```
./models/
```

#### With names:
```
G_S2T_model.hdf5
```

#### Create directories for generated images:
```
./synthetic_images/T_001
./synthetic_images/T_010
./synthetic_images/T_011
./synthetic_images/T_100
./synthetic_images/T_101
./synthetic_images/T_110
./synthetic_images/T_111
```

#### Comment row 242:
```
#self.train(…
```

#### Uncomment row 243:
```
self.load_model_and_generate_synthetic_images()
```

#### Then run:
```
python test.py
```
