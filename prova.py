# # print 10 random numbers and save them in a txt file called randon_numbers.txt
# import os
# import random

# random_numbers = [random.randint(1, 100) for _ in range(10)]

# # wait 10 seconds and then print the random numbers
# print("waiting 15 seconds")
# os.system('sleep 15')
# print("done waiting")
# print("Random numbers:")
# print(random_numbers)
# # save the random numbers in a txt file
# with open('randon_numbers.txt', 'w') as f:
#     for number in random_numbers:
#         f.write(f'{number}\n')
# print("randon_numbers.txt saved")

# total_frames = 100
# frame_num = 12 #'all'
# if frame_num != 'all' and frame_num < total_frames: # if the number of frames is less than the specified frame_num  
#     total_frames =frame_num
#     # if self.video_level:
#     #     # Select clip_size continuous frames
#     #     start_frame = random.randint(0, total_frames - self.frame_num)
#     #     frame_paths = frame_paths[start_frame:start_frame + self.frame_num]  # update total_frames
#     # else:
#     #     # Select self.frame_num frames evenly distributed throughout the video
#     #     step = total_frames // self.frame_num
#     #     frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]
#     print("using", total_frames, "frames")
# else:
#     print("using all frames")


str = '/media/data/rz_dataset/gotcha/balanced_gotcha/occlusion/testing/41/DFL/hand_occlusion/44/89455.jpg'

parts = str.split('/')  
a = parts[-2]
b = parts[-1]

print("a:", a)
print("b:", b)