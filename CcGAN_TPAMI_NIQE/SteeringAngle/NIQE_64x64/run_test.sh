python imgs_to_groups_fake.py --imgs_dir ./fake_data/fake_images --center_file ./fake_data/steering_angle_centers_loc_for_NIQE.txt --out_dir_base ./fake_data

# matlab -nodisplay -nodesktop -r "run niqe_test_steeringangle.m"
matlab -nodisplay -nodesktop -logfile output.txt -r "run niqe_test_steeringangle.m"
