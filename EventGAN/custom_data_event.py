from time import time
import numpy as np
import pathlib

# INPUT_FILE = "input/input-2-3.npy"
OUTPUT_FILE = "output/output-2-5"

THRESHOLD = .45 # measured whether percent difference between frames of light (in decimal form)
# Recommended to be between 15% to 50%

def _diff_to_polar(pixel):
    if pixel > THRESHOLD:
        return 1
    elif pixel < -THRESHOLD:
        return -1
    else:
        return 0

def polarize_frames(last_frame, curr_frame):
    """
    Take differences of frames and adjust to +1 or -1 depending on scenario
    """
    percent_diff_frame = (curr_frame - last_frame) / last_frame
    # If difference is less than one, change to -1
    polarize = np.vectorize(_diff_to_polar)
    polar_frame = polarize(percent_diff_frame)
    # diff_frame = np.where(diff_frame / 2**THRESHOLD < -1, -1, diff_frame)
    # diff_frame = np.where(diff_frame / 2**THRESHOLD > 1, 1, 0)

    return polar_frame

def format_events(frame, timestamp):
    """
    Take grid of polarities and conver it into list of polarities with elements for x and y
    """
    construct_frame = []
    for y in range(len(frame)):
        for x in range(len(frame[y])):
            
            if frame[y][x] != 0:
                construct_frame.append( [x, y, timestamp, frame[y][x]] )
    
    return construct_frame

def save_polarities(event_frames_list, i):
    # print(event_frames_list)
    # Saving Output of Polarity to directory if not exists
    pathlib.Path(OUTPUT_FILE.split("/")[0]).mkdir(exist_ok=True)
    np.save(OUTPUT_FILE+"_"+str(i)+ ".npy", event_frames_list)

def main():
    
    sim_data = np.load(INPUT_FILE, allow_pickle=True, encoding="bytes")

    image_keys = np.array(list(sim_data[()].keys()))
    k = image_keys[0]
    n_total_frames = len(image_keys)
    
    last_frame = sim_data[()][image_keys[0]]
    total_frame_list = []
    for i_key in range(1, len(image_keys)):
        if i_key % 100 == 0:
            print(f"{i_key} / {n_total_frames} - {int(i_key / n_total_frames * 100)}%")
            save_polarities(total_frame_list, i_key)
            total_frame_list = []
        # t1 = time()
        curr_frame = sim_data[()][image_keys[i_key]]
        try:
            polar_diff_frame = polarize_frames(last_frame[b"image"].astype(int), curr_frame[b"image"].astype(int))
            formatted_events = format_events(polar_diff_frame, curr_frame[b"time_stamp"])
            # print(time() - t1)
            last_frame = curr_frame
            total_frame_list.append(formatted_events)
        except KeyError:
            pass
    save_polarities(total_frame_list, i_key)
    

if __name__ == "__main__":
    main()