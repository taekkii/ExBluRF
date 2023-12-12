
datadir=./nafnet_rsblur
for scene in sunflowers camellia benchwithcamellia dragon jars laughingjars #stonelight
do
    datapath=${datadir}/${scene}
    rm -rf ${datapath}/images_0
    rm -rf ${datapath}/image_original
    rm -rf ${datapath}/raw
    rm -rf ${datapath}/sparse
    rm ${datapath}/database.db
    rm ${datapath}/poses_bounds.npy
    rm ${datapath}/test.txt
    rm ${datapath}/train.txt
    
    echo bash proc_colmap_llff.sh ${datapath} 1
    bash proc_colmap_llff.sh ${datapath} 1
done