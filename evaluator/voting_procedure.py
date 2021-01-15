"""
@ Author: ryanreadbooks
@ Time: 9/7/2020, 19:17
@ File name: voting_procedure.py
@ File description: the whole voting procedure is implemented in this file
"""
import random
import copy
from typing import List, Tuple, Dict

import numpy as np
from configs.configuration import regular_config
from utils.decorators import deprecated


class VoteProcedure(object):

    def __init__(self, img_size: Tuple, threshold: float = 0.99) -> None:
        super().__init__()
        self.threshold = threshold
        self.n_hypotheses = 50  # number of keypoint hypotheses
        self.height, self.width = img_size

    # @deprecated
    # @staticmethod
    def naive_vote(self, vmap: np.ndarray, n_out: int = 1) -> Tuple[List, np.ndarray]:
        """
        Regular voting procedure. Given a vector map of shape (2, h, w) and then return the most voted pixel position of this
        vector map.
        :param vmap: vector map of shape (2, h, w) with the order (x|y)
        :param n_out: the number of output, which is actually the top n_out votes
        :return: list of predicted keypoint (x, y) coordinates;
                votes of every selected coordinate, whose shape is (n_out, )
        """
        height, width = vmap.shape[1], vmap.shape[2]
        print('Handing naive vote')
        hough_space = np.zeros((height, width), dtype=np.int8)
        for y in range(height):
            for x in range(width):
                # go through every pixel in vector map
                nx = vmap[0, y, x]
                ny = vmap[1, y, x]
                # deal with the situation where nx=0 or ny=0
                delta = 1
                sign = 1
                x_sign = 1
                y_sign = 1
                if nx != 0:
                    delta = ny / nx
                    sign = nx / abs(nx)
                elif nx == 0:
                    x_sign = 0
                    if ny != 0:
                        sign = ny / abs(ny)
                    else:
                        y_sign = 0
                # starting from (x,y), going to the vector direction, cast a vote on hough space everywhere it goes
                x_f = x
                y_f = y
                shifted_x = x_sign * sign
                shifted_y = delta * sign * y_sign
                if shifted_x == 0 and shifted_y == 0:
                    # does not go forward, this pixel does not cast vote
                    continue
                while True:
                    x_f += shifted_x
                    y_f += shifted_y
                    x_in_hough_space = int(x_f)
                    y_in_hough_space = int(y_f)
                    if not (0 <= x_in_hough_space < width and 0 <= y_in_hough_space < height):
                        # out of bound condition
                        break
                    hough_space[y_in_hough_space, x_in_hough_space] += 1
        # voting ends, now find the max
        # sort all votes from small to large
        sorted_indices: np.ndarray = np.argsort(hough_space, None, 'mergesort')
        top_n_indices: np.ndarray = sorted_indices[-n_out:][::-1]
        # num of votes
        votes: np.ndarray = hough_space.ravel()[top_n_indices]  # shape (n_out, )
        # convert the indices back to (x, y) coordinates
        keypoint_coordinates: List = list(map(lambda arg: (arg - (arg // width) * width, arg // width), top_n_indices))

        return keypoint_coordinates, votes

    # todo re-implemente the voting process
    def ransac_vote(self, mask: np.ndarray, vmap: np.ndarray, threshold: float = 0.9) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        ransac voting procedure, which is more efficient
        :param mask: a set of coordinates that belongs to the masked object, array with shape (h, w).
            To be specific, this is a binary mask. 1 -> object, 0-> background
        :param vmap: vector map of shape (2 * NUM_KEYPOINTS, h, w) with the order (x to be the first channel | y to be the second channel)
        :param threshold: the threshold that determines whether to receive the vote, default=0.99
        :return:
        """
        eps = 1e-6
        self.threshold = threshold
        # init a dict to store the result
        score_dict: Dict[int, Tuple[np.ndarray, np.ndarray]] = dict()
        # get the population

        population = np.where(mask[0][0] >= 1)[:2]
        population: List[Tuple] = list(zip(population[1], population[0]))  # the set of coordinates, format List[(x,y)]
        center = np.asarray(population).mean(axis=0)    # consider it as the center of the mask
        print('center', center)

        # process every keypoint
        assert (vmap.shape[0] / 2 == regular_config.num_keypoint), 'number of keypoints does not match'
        for i in range(regular_config.num_keypoint):
            v_k = vmap[i * 2: (i + 1) * 2]  # shape of (2, h, w)
            candidates: np.ndarray = self._generate_candidates_from_population(population, v_k, threshold)  # (self.n_hypotheses, 2)
            # init a space to store the votes the candidates get
            candidates_votes: np.ndarray = np.zeros((self.n_hypotheses, 1))
            # init a list to store the candidates and their scores
            candidates_copy: np.ndarray = copy.deepcopy(candidates)
            # the rest of the population takes part in the voting
            for voter in population:
                # voter's location
                voter_x, voter_y = voter[0], voter[1]
                candidates_copy[:, 0] -= voter_x
                candidates_copy[:, 1] -= voter_y
                norm = np.linalg.norm(candidates_copy, axis=1).reshape((-1, 1))
                candidates_unit_vector = candidates_copy / (norm + eps)  # shape of (self.n_hypotheses, 2)
                # voter vector here is not unit maybe, due to the incorrectness of the neural network, so we manually make it a unit vector
                voter_vector = v_k[:, voter_y, voter_x].reshape((2, 1))
                voter_vector /= np.linalg.norm(voter_vector)
                scores = candidates_unit_vector @ voter_vector  # shape of (self.n_hypotheses, 1)
                # print('scores: \n', scores)
                candidates_votes += (scores >= self.threshold)
            # one keypoint has been processed, update the score_dict
            score_dict[i] = (candidates, candidates_votes)
        # print('score_dict of all hypotheses: \n', score_dict)
        return score_dict

    def _vote_average(self, score: Dict, weighted=True) -> Dict[int, np.ndarray]:
        """
        calculate the mean position of ransac voting results
        :param score: result from ransac voting
        :param weighted: use weighted or not
        :return: average keypoints
        """
        result_keypoint = dict()
        for key, value in score.items():
            candidates, votes = value[0], value[1]
            if weighted:
                total_votes = np.sum(votes)
                if total_votes != 0:
                    weights = (votes / total_votes).reshape(-1)
                    result = np.average(candidates, axis=0, weights=weights)
                else:
                    print('Total is zero, now switching into the mean mode')
                    result = np.mean(candidates, axis=0)
            else:
                result = np.mean(candidates, axis=0)
            result_keypoint[key] = result
        return result_keypoint

    def _vote_covariance(self, score: Dict) -> Dict[int, np.ndarray]:
        """
        calculate the mean position of ransac voting results
        :param score: result from ransac voting
        :return: covariance of the keypoints
        """
        pass

    def _generate_candidates_from_population(self, population: List, vectors: np.ndarray, threshold) -> np.ndarray:
        """
        choose the keypoint candidates from the population
        :param population: the population which would choose their candidates, List[(x, y)]
        :param vectors: vector map of shape (2, h, w)
        :return: chosen candidates
        """
        candidates = list()
        assert len(population) != 0, 'population has been exhausted.'
        cnt = 0
        # print('Generating candidates...')
        while True:

            # pop element here will affect the list outside this method
            point1: Tuple = population.pop(random.randrange(len(population)))  # (x1, y1)
            point2: Tuple = population.pop(random.randrange(len(population)))  # (x2, y2)
            # the corresponding vectors
            vector1: np.ndarray = vectors[:, point1[1], point1[0]]  # shape (2, ), format (vx1, vy1)
            vector2: np.ndarray = vectors[:, point2[1], point2[0]]  # shape (2, ), format( vx2, vy2)
            # find their intersection and make the intersection a candidate
            slope1 = vector1[1] / vector1[0]
            slope2 = vector2[1] / vector2[0]
            # check if two lines parallel
            if not (slope1 == slope2):
                intercept1 = point1[1] - point1[0] * slope1
                intercept2 = point2[1] - point2[0] * slope2
                # until now, we already know the equation of two lines
                # find the intersection point between two lines
                intersection_x = (intercept1 - intercept2) / (slope2 - slope1)
                intersection_y = slope1 * intersection_x + intercept1
                # check if the coordinate of intersection is valid (the same direction of unit vector and not out of bound)
                valid = self.check_intersection_validity(point1, point2, vector1, vector2, intersection_x, intersection_y, threshold)
                if valid:
                    intersection = np.array([intersection_x, intersection_y])
                    candidates.append(intersection)
            # authorize them to vote again
            population.append(point1)
            population.append(point2)
            if len(candidates) == self.n_hypotheses:
                break

        # print('Generating candidates done...')

        return np.array(candidates)  # shape (self.n_hypotheses, 2) of format (x, y)

    def _remove_outliers(self, keypoints: Dict, std_factor: int = 2) -> Dict:
        """
        remove the outliers in keypoint hypotheses
        :param keypoints: keypoint hypotheses, dict containing keypoint coordinates and keypoints votes
        :return: keypoints hypotheses without outliers. Dict containing keypoint coordinates and keypoints votes
        """
        inlier_kps_dict: Dict[int, Tuple[np.ndarray, np.ndarray]] = dict()
        for key, value in keypoints.items():
            kps, votes = value
            kps_mean = np.mean(kps, axis=0)
            kps_std = np.std(kps, axis=0)
            # range = [mean - factor * std, mean + factor * std]
            kps_range_max: np.ndarray = kps_mean + std_factor * kps_std
            kps_range_min: np.ndarray = kps_mean - std_factor * kps_std
            qualified_indices_x: np.ndarray = np.array((kps[:, 0] >= kps_range_min[0]) & (kps[:, 0] <= kps_range_max[0]))
            qualified_indices_y: np.ndarray = np.array((kps[:, 1] >= kps_range_min[1]) & (kps[:, 1] <= kps_range_max[1]))
            inlier_indices: np.ndarray = qualified_indices_x * qualified_indices_y
            if inlier_indices.sum() == 0:
                # if no inliers found, then we'll have to treat all points are inliers
                inlier_indices = np.array([True] * kps.shape[0])
            inlier_kps = kps[inlier_indices]
            inlier_kps_votes = votes[inlier_indices]
            inlier_kps_dict[key] = (inlier_kps, inlier_kps_votes)

        return inlier_kps_dict

    def provide_keypoints(self, mask: np.ndarray, vmap: np.ndarray, threshold: float = 0.9, weighted=True) -> np.ndarray:
        """
        the combination of ransac voting and vote averaging
        ransac voting procedure, which is more efficient
        :param mask: a set of coordinates that belongs to the masked object, shape (h, w)
        :param vmap: vector map of shape (2 * NUM_KEYPOINTS, h, w) with the order (x to be the first channel | y to be the second channel)
        :param threshold: the threshold that determines whether to receive the vote, default=0.99
        :param weighted: use weighted or not
        :return: final keypoints, np.ndarray of shape (num_keypoints, 2)
        """
        # ransac voting the generate hypotheses
        temp_keypoints: Dict = self.ransac_vote(mask=mask, vmap=vmap, threshold=threshold)
        # remove some outliers in the keypoints hypotheses
        temp_keypoints: Dict = self._remove_outliers(temp_keypoints)
        # compute the mean of the remaining keypoints
        keypoints_dict: Dict[int, np.ndarray] = self._vote_average(temp_keypoints, weighted)
        # print(keypoints_dict)
        keypoints = np.array(list(keypoints_dict.values()))
        # print('Keypoint provided...')
        return keypoints

    def check_intersection_validity(self, p1: Tuple, p2: Tuple, v1: np.ndarray, v2: np.ndarray, x: float, y: float, threshold) -> bool:
        """
        check if the intersection is valid
        :param p1: point1
        :param p2: point2
        :param v1: unit vector of point1
        :param v2: unit vector of point2
        :param x: intersection x coordinate
        :param y: intersection y coordinate
        :return: valid - True, invalid - False
        """
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            # intersection is out of image bound
            # print('out of bound invalid')
            return False
        else:
            # check if the intersection is at the direction of v1 and v2
            d1 = np.array([x - p1[0], y - p1[1]])
            d2 = np.array([x - p2[0], y - p2[1]])
            # todo double check here
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            if d1 @ v1.reshape((2, 1)) >= threshold and d2 @ v2.reshape((2, 1)) >= threshold:
                return True
        return False


# todo: try to implement the ransac voting in GPU, using numba.cuda
def ransac_voting_gpu():
    raise NotImplementedError


# module testing
if __name__ == '__main__':
    width1, height1 = 500, 500
    epss = 1e-6
    x1 = np.linspace(0, width1 - 1, width1)
    y1 = np.linspace(0, height1 - 1, height1)
    xx1, yy1 = np.meshgrid(x1, y1)
    pixel_coordinate_map = np.concatenate([xx1[np.newaxis], yy1[np.newaxis]], axis=0)
    keypoint = np.array([2, 3]).reshape((2, 1, 1))  # (x coordinate, y coordinate)
    dif = keypoint - pixel_coordinate_map
    dif_norm = np.linalg.norm(dif, axis=0)
    temp_vmap = dif / (dif_norm + epss)
    temp_vmap = np.repeat(temp_vmap, 9, axis=0)
    print('vmap shape', temp_vmap.shape)
    # test ransac voting
    ma = np.random.rand(width1, height1) * 5
    ma = np.where(ma > 1.5, 1, 0)
    print('mask shape', ma.shape)
    procedure = VoteProcedure((height1, width1))
    d = procedure.ransac_vote(ma, temp_vmap)
    print(d)
    print(procedure._vote_average(d))
