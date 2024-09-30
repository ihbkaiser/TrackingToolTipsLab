import torch
from base64 import b64encode
from IPython.display import HTML
import imageio.v3 as iio
import matplotlib.pyplot as plt
import supervision as sv
from datetime import datetime
from inference_sdk import InferenceHTTPClient
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
online_model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online").to(device)
offline_model = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)
segm_model = InferenceHTTPClient(api_url="https://outline.roboflow.com",api_key="NzQOXBqQrzYP4mT38Ksg")
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor
import cv2
import numpy as np
from skimage.morphology import skeletonize
from tqdm import tqdm
def get_length(video_path):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length
def get_endpoints(skeleton_mask):
    bp = np.where(skeleton_mask !=0)
    nonzero_coord = list(zip(bp[0],bp[1]))
    endpoints = []
    for coord in nonzero_coord: 
        row, col = coord 
        rows, cols = skeleton_mask.shape
        start_row = max(0, row - 1)
        end_row = min(rows, row + 2)
        start_col = max(0, col - 1)
        end_col = min(cols, col + 2)
        neighbors = skeleton_mask[start_row:end_row, start_col : end_col]
        if(np.count_nonzero(neighbors)==2):
            endpoints.append(coord)
    return endpoints

def view_points(coords, image, save_dir=None):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for coord in coords:
        plt.scatter(coord[1], coord[0], color='red', s=5)  # Flipping row and col for plotting
    if save_dir is None:
        plt.imshow(image, cmap='gray')
        plt.show()
    else:
        plt.imshow(image, cmap='gray')
        plt.axis('off')  # Tắt trục
        plt.savefig(save_dir, bbox_inches='tight', pad_inches=0)
        plt.close()


def display_video(path_to_video):
    with open(path_to_video, "rb") as file:
        video_data = file.read()
    encoded_video = b64encode(video_data).decode()
    video_url = f"data:video/mp4;base64,{encoded_video}"
    video_html = f"""
        <video width="640" height="480" autoplay loop controls>
            <source src="{video_url}" type="video/mp4">
        </video>
    """
    return HTML(video_html)

def extract_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame



def get_from_roboflow(model_id, image):
    result = segm_model.infer(image, model_id)
    return result

def get_sv_detections(image):
    result = get_from_roboflow("sugical-tool-iszjm/1", image)
    detections = sv.Detections.from_inference(result)
    return detections
def get_mask(image):
    result = get_from_roboflow("sugical-tool-iszjm/1", image)
    detections = sv.Detections.from_inference(result)
    return detections.mask


# # Tạo một frame dạng np.array ví dụ


# selected_points = []
# def onclick(event):
#     if event.xdata is not None and event.ydata is not None:
#         x = int(round(event.xdata))
#         y = int(round(event.ydata))
#         print('Tọa độ Đề các:', x, y)
#         selected_points.append((x, y))
#         update_plot()


# def update_plot():
#     plt.imshow(frame, cmap='gray')
#     plt.title('Tọa độ Đề các')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.gca().set_aspect('equal', adjustable='box')  # Đảm bảo tỷ lệ trục x và y là bằng nhau
#     plt.grid(True)
#     for point in selected_points:
#         plt.plot(point[0], point[1], 'ro')  # Tô màu đỏ cho điểm đã chọn
#     plt.gcf().canvas.draw()

# def analysis(frame):
#     selected_points = []
#     plt.imshow(frame, cmap='gray')
#     plt.title('Tọa độ Đề các')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.gca().set_aspect('equal', adjustable='box')  
#     plt.grid(True)
#     plt.gcf().canvas.mpl_connect('button_press_event', onclick)
#     plt.show()


# video_path = 'D:/tool_test_01.mp4'
# first_frame = extract_frame(video_path, 0)
# analysis(first_frame)

def get_starting_point(abstract_point, points):
    dist = -np.inf
    starting_point = None
    for point in points:
        if dist < np.linalg.norm(np.array(abstract_point) - np.array(point)):
            dist = np.linalg.norm(np.array(abstract_point) - np.array(point))
            starting_point = point
    return starting_point
def get_tip(abstract_point, points):
    starting_point = get_starting_point(abstract_point, points)
    dist = -np.inf
    tip_point = None
    for point in points:
        if dist < np.linalg.norm(np.array(starting_point) - np.array(point)):
            dist = np.linalg.norm(np.array(starting_point) - np.array(point))
            tip_point = point
    return tip_point
    print(tip_point)

def _process_step(model,window_frames, is_first_step, grid_size, grid_query_frame, queries, segm_mask):
        video_chunk = (
            torch.tensor(np.stack(window_frames[-model.step * 2 :]), device=device)
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            queries = queries,
            segm_mask = segm_mask
        )
def model_online_processing(video_path, model = online_model, grid_size=None, grid_query_frame= 0 , queries=None, segm_mask=None):
    window_frames = []
    is_first_step = True
    for i, frame in enumerate(iio.imiter(video_path,plugin="FFMPEG",)):
        if i % model.step == 0 and i != 0:
            # print(torch.tensor(np.stack(window_frames[-model.step * 2 :])).shape)
            pred_tracks, pred_visibility = _process_step(
                    model,
                    window_frames,
                    is_first_step,
                    grid_size=grid_size,
                    grid_query_frame=grid_query_frame,
                    queries = queries,
                    segm_mask = segm_mask
                )
            is_first_step = False
        window_frames.append(frame)
    pred_tracks, pred_visibility = _process_step(
        model,
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        grid_size=grid_size,
        grid_query_frame=grid_query_frame,
        queries = queries,
        segm_mask = segm_mask
    )
    return pred_tracks, pred_visibility

def add_zero_column(matrix):
    result_matrix = np.column_stack([np.zeros(matrix.shape[0]), matrix])
    return result_matrix

def skeleton_query(skeleton_points):
    numpy_queries = add_zero_column(skeleton_points)
    torch_queries = torch.from_numpy(numpy_queries)
    column1 = torch_queries[:, 1].clone().detach()
    column2 = torch_queries[:, 2].clone().detach()
    torch_queries[:,1]=column2 
    torch_queries[:,2]=column1
    return torch_queries.unsqueeze(0).to(device).float()
def reshape_video(video):
    return torch.from_numpy(video).permute(0,3,1,2)[None].float().to(device)

def get_idx(point_qt, points):  #points must be in point_qt, else error.
    idx_qt = np.where(  (points[:,0] == point_qt[0])  &   (points[:,1] == point_qt[1]))[0][0]
    return idx_qt 

def show_skeleton(binary_mask, image):
    binary_mask_color = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    binary_mask_color = cv2.resize(binary_mask_color, (image.shape[1], image.shape[0]))

    result = cv2.addWeighted(image, 1, binary_mask_color, 1, 0)

    return result
def analysis(mask, abstract_point):
    idx_qt = []
    try:
        skeleton_pool = [np.argwhere(skeletonize(mask[i])) for i in range(len(mask))]
    except:
        skeleton_pool = []
    skeleton_points = np.vstack(skeleton_pool)
    for i in range(len(mask)):
        tip = get_tip( abstract_point, skeleton_pool[i]) 
        tip_idx = get_idx(tip, skeleton_points)
        idx_qt.append(tip_idx) 
    return idx_qt, skeleton_query(skeleton_points)

#view skeletonize mask on an image


def remove_black_pixels(frame1, threshold):  #threshold: the pixel may be black (0) or almost-black (ex:0,1,2,3,...)
    frame = frame1.copy()
    # Tạo một mask để xác định các pixel màu đen
    black_mask = np.all(frame >= 0, axis=-1) & np.all(frame <= threshold, axis=-1)
    
    # Gán các pixel màu đen thành màu trắng (255, 255, 255)
    frame[black_mask] = [255, 255, 255]
    
    return frame
 
def Testing_circle(image, abstract_point, save_dir = None, save_error = None):
    result = get_from_roboflow("sugical-tool-iszjm/1", image)
    detections = sv.Detections.from_roboflow(result)
    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image, detections=detections)
    mask = detections.mask
    if mask is not None:
            skeleton = skeletonize(mask)                
        # change skeleton mask to list of points
            tips = []
            if len(skeleton.shape) == 2:
                rows, cols = skeleton.shape
                skeleton = skeleton.reshape(1, rows, cols)
            for i in range(len(skeleton)):
                annotated_image = show_skeleton(skeleton[i], annotated_image)
                endpoints = get_endpoints(skeleton[i])
                tip = get_tip(abstract_point, endpoints)
                if tip is not None:
                    tips.append(tip)
            image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            view_points(tips, image, save_dir)
    else:
        cv2.imwrite(save_error, annotated_image)

def Testing_rectangle_video(video_path, num_frames, ROI, save_dir = None, save_error = None):
    frame_generator = sv.get_video_frames_generator(video_path)
    video_info = sv.VideoInfo.from_video_path(video_path)
    tracker = sv.ByteTrack(frame_rate=video_info.fps, track_thresh=0.3)
    for idx, frame in tqdm(enumerate(frame_generator)):
        frame = crop(frame, ROI)
        result = get_from_roboflow("sugical-tool-iszjm/1", frame)
        detections = sv.Detections.from_roboflow(result)
        mask_annotator = sv.MaskAnnotator()
        annotated_image = mask_annotator.annotate(scene=frame, detections=detections)
        mask = detections.mask
        bounding_boxes = detections.xyxy
        detections = tracker.update_with_detections(detections)
        if mask is not None:
            print(f'Frame {idx} : OK')
            skeleton = skeletonize(mask)
            # change skeleton mask to list of points
            tips = []
            if len(skeleton.shape) == 2:
                rows, cols = skeleton.shape
                skeleton = skeleton.reshape(1, rows, cols)
            for i in range(len(skeleton)):
                annotated_image = show_skeleton(skeleton[i], annotated_image)
                abstract_edge = get_abstract_edge(bounding_boxes[i], ROI)
                endpoints = get_endpoints(skeleton[i])
                if len(endpoints) > 0:
                    handle = get_handle(abstract_edge, endpoints)
                    tip = get_tip(handle, endpoints)
                    if tip is not None:
                        tips.append(tip)
            image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            view_points(tips, image, save_dir + str(idx) + ".png")
        else:
            print(f'Frame {idx} : Error')
            cv2.imwrite(save_error + str(idx) + ".png", annotated_image)
        if(idx == num_frames):
            break

def Testing_rectangle(image, ROI, save_dir = None, save_error = None):
    
    image = crop(image, ROI)
    result = get_from_roboflow("sugical-tool-iszjm/1", image)
    detections = sv.Detections.from_roboflow(result)
    # detections = tracker.update_with_detections(detections)
    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image, detections=detections)
    mask = detections.mask
    bounding_boxes = detections.xyxy
    if mask is not None:
            skeleton = skeletonize(mask)
            # change skeleton mask to list of points
            tips = []
            if len(skeleton.shape) == 2:
                rows, cols = skeleton.shape
                skeleton = skeleton.reshape(1, rows, cols)
            for i in range(len(skeleton)):
                annotated_image = show_skeleton(skeleton[i], annotated_image)
                abstract_edge = get_abstract_edge(bounding_boxes[i], ROI)
                endpoints = get_endpoints(skeleton[i])  #TODO: endpoints can be empty ??
                if len(endpoints) > 0:
                    handle = get_handle(abstract_edge, endpoints)
                    tip = get_tip_rec(handle, endpoints)
                    if tip is not None:
                        tips.append(tip)
            image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            view_points(tips, image, save_dir)
            return "Success run"
    else:
        cv2.imwrite(save_error, annotated_image)
        return "No mask"



class TrackModel:
    def __init__(self, video_path, grid_size=None, grid_query_frame= 0 , queries=None, segm_mask=None):
        self.video_path = video_path
        self.model = online_model
        self.grid_size = grid_size
        self.grid_query_frame = grid_query_frame
        self.queries = queries
        self.segm_mask = segm_mask
        self.pred_tracks = None 
        self.pred_visibility = None
    def run(self):
        self.pred_tracks, self.pred_visibility = model_online_processing(self.video_path, self.model, self.grid_size, self.grid_query_frame, self.queries, self.segm_mask)
    def show(self, save_path, video_path, idx_qt = None):
        vis = Visualizer(save_dir = save_path, linewidth = 6, mode='cool', tracks_leave_trace=-1)
        if idx_qt is not None and type(idx_qt)== int:
            vis.visualize(reshape_video(read_video_from_path(video_path)), self.pred_tracks[:,:,idx_qt,:].unsqueeze(-2), self.pred_visibility[:,:,idx_qt].unsqueeze(-1) )
        elif idx_qt is not None and type(idx_qt)== list:
            vis.visualize(reshape_video(read_video_from_path(video_path)), self.pred_tracks[:,:,idx_qt,:], self.pred_visibility[:,:,idx_qt] )
        
        else:
            vis.visualize(reshape_video(read_video_from_path(video_path)), self.pred_tracks , self.pred_visibility)
    def quick_show(self):
        _, T, N, _ = self.pred_tracks.shape 
        cap = cv2.VideoCapture(self.video_path)
        for i in range(T):
            frame = cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            for j in range(N):
                x,y = self.pred_tracks[0,i,j,0], self.pred_tracks[0,i,j,1]
                if self.pred_visibility[0,i,j] == 0:
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)
                if self.pred_visibility[0,i,j] == 1:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

##################### MORE FUNCTIONS FOR RECTANGLE VIDEO ############################
def get_handle(abstract_edge, points):
    # idx, coord = abstract_edge
    # if idx in [0,1]:
    #     min_idx = np.argmin([abs(point[1] - coord) for point in points])
    # if idx in [2,3]:
    #     min_idx = np.argmin([abs(point[0] - coord) for point in points])
    # return points[min_idx]
    assert len(abstract_edge) <=2
    if len(abstract_edge) == 1:
        idx, coord = abstract_edge[0]
        if idx in [0,1]:
            min_idx = np.argmin([abs(point[1] - coord) for point in points])
        if idx in [2,3]:
            min_idx = np.argmin([abs(point[0] - coord) for point in points])
        return points[min_idx]
    if len(abstract_edge) == 2:
        edge_point = [abstract_edge[1][1], abstract_edge[0][1]]
        min_idx = np.argmin([np.linalg.norm(np.array(edge_point) - np.array(point)) for point in points])
        return points[min_idx]
def get_tip_rec(handle, points):
    dist = -np.inf
    tip_point = None
    try:
        for point in points:
            current_dist = np.linalg.norm(np.array(handle) - np.array(point))
            if dist < current_dist:
                dist = current_dist
                tip_point = point
    except:
        pass
    return tip_point

def get_abstract_edge(bounding_box, ROI, threshold):
    x1 = bounding_box[0]
    x2 = bounding_box[2]
    y1 = bounding_box[1]
    y2 = bounding_box[3]
    X1 = 0
    X2 = ROI[0][1]-ROI[0][0]
    Y1 = 0
    Y2 = ROI[1][1]-ROI[1][0]
    ####################### First strategy ###########################
    # edge_idx = np.argmin([abs(x1 - X1), abs(x2 - X2), abs(y1 - Y1), abs(y2 - Y2)]) #left, right, top, down
    # edge_num = [x1, x2, y1, y2][edge_idx]
    # abstract_edge = [edge_idx, edge_num]
    ###################### Second strategy ###########################
    distance_list = [abs(x1 - X1), abs(x2 - X2), abs(y1 - Y1), abs(y2 - Y2)]
    abstract_edge = []
    if abs(x1 - X1) < threshold*X2:
        abstract_edge.append([0, x1])
    if abs(x2 - X2) < threshold*X2:
        abstract_edge.append([1, x2])
    if abs(y1 - Y1) < threshold*Y2:
        abstract_edge.append([2, y1])
    if abs(y2 - Y2) < threshold*Y2:
        abstract_edge.append([3, y2])
    return abstract_edge

def crop(image, ROI): #expect ROI is [[x1, x2], [y1, y2]], we want to crop x1<=X<=x2, y1<=Y<=y2
    x1 = ROI[0][0]
    x2 = ROI[0][1]
    y1 = ROI[1][0]
    y2 = ROI[1][1]
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def get_bb(image):
    result = get_from_roboflow("sugical-tool-iszjm/1", image)
    detections = sv.Detections.from_inference(result)
    return detections.xyxy


