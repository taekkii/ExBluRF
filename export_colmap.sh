INPUT_DIR=$1
colmap feature_extractor  --database_path $INPUT_DIR/db.db --image_path $INPUT_DIR/images_train --ImageReader.single_camera 1 --ImageReader.camera_model SIMPLE_PINHOLE --SiftExtraction.use_gpu 0 --ImageReader.camera_params "$2,$3,$4"
colmap exhaustive_matcher --database_path $INPUT_DIR/db.db  --SiftMatching.guided_matching 1 --SiftMatching.use_gpu 0
mkdir -p $INPUT_DIR/sparse/0
python run_colmap/reorder_db.py --datadir $INPUT_DIR
colmap point_triangulator --database_path $INPUT_DIR/db.db --image_path $INPUT_DIR/images_train --input_path $INPUT_DIR/sparse_learned --output_path $INPUT_DIR/sparse/0 #--Mapper.abs_pose_min_inlier_ratio 0.15 --Mapper.abs_pose_min_num_inliers 10

colmap feature_extractor --database_path $INPUT_DIR/db.db --image_path $INPUT_DIR/images --image_list_path $INPUT_DIR/train.txt --SiftExtraction.use_gpu 0 --ImageReader.single_camera=1 --ImageReader.camera_model=SIMPLE_PINHOLE --ImageReader.camera_params "$2,$3,$4"
colmap vocab_tree_matcher --database_path $INPUT_DIR/db.db --VocabTreeMatching.vocab_tree_path ./run_colmap/vocab_tree_flickr100K_words32K.bin --VocabTreeMatching.match_list_path $INPUT_DIR/train.txt --SiftMatching.use_gpu 0
mkdir -p $INPUT_DIR/sparse/1
colmap image_registrator --database_path $INPUT_DIR/db.db --input_path $INPUT_DIR/sparse/0 --output_path $INPUT_DIR/sparse/1

mv $INPUT_DIR/sparse/0 $INPUT_DIR/sparse/0_trainset
mv $INPUT_DIR/sparse/1 $INPUT_DIR/sparse/0

python run_colmap/colmap2llff.py --datadir $1
