from ROI import ROI
from cotracker_utils import get_abstract_edge, get_handle
import numpy as np
import networkx as nx
class Image:
    def __init__(self, image, whiten_threshold = 10, use_roboflow = True):
        self.image = image
        self.bounding_box = None
        self.masks = None
        self.skeletons = None
        self.already_set = False
        self.threshold = whiten_threshold
    def whiten(self, white_pixel = 255):
        self.image[self.image > self.threshold] = white_pixel

    def shape(self):
        h, w = self.image.shape[:2]
        return (h, w)
    def crop(self, range_of_interest = None):
        if range_of_interest is not None:
            x1 = range_of_interest[0][0]
            x2 = range_of_interest[0][1]
            y1 = range_of_interest[1][0]
            y2 = range_of_interest[1][1]
            self.image = self.image[y1:y2, x1:x2]
        else:
            self.image = self.image[ROI[1][0]:ROI[1][1], ROI[0][0]:ROI[0][1]]
    def setBoundingBox(self, model):
        self.bounding_box = model.getBoundingBox(self.image)
    def setMasks(self, model):
        self.masks = model.getMasks(self.image)
    def setSkeletons(self):
        self.skeletons = [m.getSkeleton() for m in self.masks]
    def setAll(self, model):
        self.setBoundingBox(model)
        self.setMasks(model)
        self.setSkeletons()
        self.already_set = True
    def show(self):
        sv.plot_image(self.image)
    # def showAll(self):
    #     # raise error if setAll() was not called
    #     if not self.already_set:
    #         raise ValueError("setAll() must be called before showAll(), else nothing to show")
    #     annotated_frame = self.image.copy()
    #     for i in range(len(self.bounding_box)):
    #         x1, y1, x2, y2 = self.bounding_box[i]
    #         # draw rectangle x1 <= x <= x2, y1 <= y <= y2
    #         pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
    #         pts = pts.reshape((-1, 1, 2))
    #         annotated_frame = cv2.polylines(annotated_frame, [pts], True, (0, 255, 0), 2)
    #     # Draw masks
    #     for m in self.masks:
    #         mask = np.array(m.mask, dtype=np.uint8)
    #         color_mask = np.dstack((mask, mask, mask)) * 255  # Create a 3-channel image with the same mask in all channels
    #         color_mask = color_mask.astype(annotated_frame.dtype)  # Convert color_mask to the same data type as annotated_frame
    #         annotated_frame = cv2.addWeighted(color_mask, 0.3, annotated_frame, 0.7, 0)
    #     # Draw skeletons
    #     for s in self.skeletons:
    #         # change s from bool matrix to 0-1 matrix
    #         s = np.array(s, dtype=np.uint8)
    #         skeleton_image = np.where(s == 1, 0, 255)  # Create an image where the pixels with value 1 in s are black and the rest are white
    #         skeleton_image = np.dstack((skeleton_image, skeleton_image, skeleton_image))  # Convert to 3-channel image
        #     skeleton_image = skeleton_image.astype(annotated_frame.dtype)  # Convert skeleton_image to the same data type as annotated_frame
        #     annotated_frame = cv2.addWeighted(skeleton_image, 0.3, annotated_frame, 0.7, 0)

        # sv.plot_image(annotated_frame)
    def showBoundingBox(self):
        assert self.bounding_box is not None , "Bounding box not set"
        image_with_boxes = self.image.copy()
        for box in self.bounding_box:
            # each box is (x1, y1, x2, y2), draw x1 <= X <= x2, y1 <= Y <= y2
            x1, y1, x2, y2 = box
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2))
        sv.plot_image(image_with_boxes)
    def showMask(self):
        assert self.masks is not None, "Mask not set"
        image_with_mask = self.image.copy()
        for mask in self.masks:
            _mask = mask.mask
            # each mask is a H x W image, of black and white resprent the mask
            # plot the mask with red color on the image_with_mask
            _mask = np.dstack((_mask, np.zeros_like(_mask), np.zeros_like(_mask)))
            _mask = _mask.astype(image_with_mask.dtype)
            image_with_mask = cv2.addWeighted(_mask, 0.9, image_with_mask, 0.1, 0)
        sv.plot_image(image_with_mask)
    def showSkeleton(self):
        assert self.skeletons is not None, "Skeleton not set"
        image_with_skeleton = self.image.copy()
        for skeleton in self.skeletons:
            skeleton_image = np.dstack((skeleton, np.zeros_like(skeleton), np.zeros_like(skeleton)))  # Convert to 3-channel image
            skeleton_image = skeleton_image.astype(image_with_skeleton.dtype)  # Convert skeleton_image to the same data type as image_with_skeleton
            image_with_skeleton = cv2.addWeighted(skeleton_image, 0.3, image_with_skeleton, 0.7, 0)
        sv.plot_image(image_with_skeleton)
        
    def getIns(self, dijkstra = True, leftright_margin_threshold = 5, updown_margin_threshold = 0, abstract_edge_threshold = 0.1, range_of_interest = ROI):
        '''
        Assume the image has M skeletons, then this function returns a list contains M lists,
        each list contains the tip of the corresponding skeleton
        Args:
            leftright_margin_threshold (LMT): Say the image is x1 <= X <= x2, y1 <= Y <= y2, then we only consider x1 + LMT <= X <= x2 - LMT
            updown_margin_threshold (UMT): Say the image is x1 <= X <= x2, y1 <= Y <= y2, then we only consider y1 + UMT <= Y <= y2 - UMT
            abstract_edge_threshold : An edge is considered close to the boundary ,
            if the distance d between that edge and the boundary is less than abstract_edge_threshold * width (or length)
            range_of_interest : ROI of frame.
            dijkstra: If True, use Dijkstra to find the tip point from the handle using accumulated distance, else use normal Euclidean distance
        '''
        assert self.masks is not None, "Mask not set"
        tips = []
        handles = []
        endpoints = []

        for idx, skeleton in enumerate(self.skeletons):
              img_array = np.array(skeleton)
              G = nx.Graph()
              for i in range(img_array.shape[0]):
                  for j in range(img_array.shape[1]):
                      if img_array[i, j] == 1:
                          G.add_node((i, j))
              for i in range(img_array.shape[0]):
                  for j in range(img_array.shape[1]):
                      # check margin condition
                      if i > updown_margin_threshold and i < img_array.shape[0] - updown_margin_threshold and j > leftright_margin_threshold and j < img_array.shape[1] - leftright_margin_threshold:
                          if img_array[i, j] == 255:
                              G.add_node((i,j))


              for node in G.nodes:
                  # Check neighbors
                  for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, -1), (1, 1)]:
                      neighbor = (node[0] + dx, node[1] + dy)
                      # If the neighbor is also a white pixel, add corresponding edge
                      if neighbor in G.nodes:
                          G.add_edge(node, neighbor, weight = np.sqrt(dx**2 + dy**2))
              # Endpoints are nodes with degree of 1
              node_list = [node for node in G.nodes]
              end_points = [node for node in G.nodes if G.degree(node) == 1]
              abstract_edge = get_abstract_edge(self.bounding_box[idx], range_of_interest, abstract_edge_threshold)
              # handle = get_handle(abstract_edge, end_points)

              # if G has no nodes, then we dont care
              if len(node_list) == 0:
                  continue
              try:
                  handle = get_handle(abstract_edge, node_list)
              except:
                  print("Error node_list")
              # path_lengths =  nx.single_source_dijkstra_path_length(G, handle)
              # tip = max(path_lengths, key=path_lengths.get)
              endpoints.append(end_points)
              handles.append(handle)
              if not dijkstra:
                dist = np.linalg.norm(np.array(node_list) - np.array(handle), axis = 1)
                farthest_index = np.argmax(dist)
                tips.append(node_list[farthest_index])
              else:
                path_lengths =  nx.single_source_dijkstra_path_length(G, handle)
                tip = max(path_lengths, key=path_lengths.get)
                tips.append(tip)
        return tips, handles, endpoints