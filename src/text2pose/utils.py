import torch

################################################################################
## READ/WRITE TO FILES
################################################################################

import json

def save_to_file(data, filename_res):
    with open(filename_res, "w") as f:
        f.write(json.dumps(data))
        print("Saved file:", filename_res)

def load_from_file(filename_res):
    with open(filename_res, "r") as f:
        data = json.load(f)
        data = {k:float(v) for k, v in data.items()} # parse values
        print("Load results from", filename_res)
    return data


################################################################################
## MISCELLANEOUS CALCULATION
################################################################################

from scipy import linalg

def mean_list(data):
    return sum(data)/len(data)

def mean_std_list(data):
    m = mean_list(data)
    s = sum((x-m)**2 for x in data)/len(data)
    return [m, s**0.5]

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


################################################################################
## LOGGING
################################################################################

import datetime
import time
from collections import defaultdict, deque


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a window
    or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


################################################################################
## ANGLE TRANSFORMATION FONCTIONS
################################################################################

import roma

def rotvec_to_eulerangles(x):
    x_rotmat = roma.rotvec_to_rotmat(x)
    thetax = torch.atan2(x_rotmat[:,2,1], x_rotmat[:,2,2])
    thetay = torch.atan2(-x_rotmat[:,2,0], torch.sqrt(x_rotmat[:,2,1]**2+x_rotmat[:,2,2]**2))
    thetaz = torch.atan2(x_rotmat[:,1,0], x_rotmat[:,0,0])
    return thetax, thetay, thetaz

def eulerangles_to_rotmat(thetax, thetay, thetaz):
    N = thetax.numel()
    # rotx
    rotx = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
    roty = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
    rotz = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
    rotx[:,1,1] = torch.cos(thetax)
    rotx[:,2,2] = torch.cos(thetax)
    rotx[:,1,2] = -torch.sin(thetax)
    rotx[:,2,1] = torch.sin(thetax)
    roty[:,0,0] = torch.cos(thetay)
    roty[:,2,2] = torch.cos(thetay)
    roty[:,0,2] = torch.sin(thetay)
    roty[:,2,0] = -torch.sin(thetay)
    rotz[:,0,0] = torch.cos(thetaz)
    rotz[:,1,1] = torch.cos(thetaz)
    rotz[:,0,1] = -torch.sin(thetaz)
    rotz[:,1,0] = torch.sin(thetaz)
    rotmat = torch.einsum('bij,bjk->bik', rotz, torch.einsum('bij,bjk->bik', roty, rotx))
    return rotmat

def eulerangles_to_rotvec(thetax, thetay, thetaz):
    rotmat = eulerangles_to_rotmat(thetax, thetay, thetaz)
    return roma.rotmat_to_rotvec(rotmat)


################################################################################
## LOAD POSE DATA
################################################################################

import os
import numpy as np

import text2pose.config as config


def read_posescript_json(relative_filepath):
    filepath = os.path.join(config.POSESCRIPT_LOCATION, relative_filepath)
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def get_pose_data_from_file(pose_info):
    """
    Load pose data and normalize the orientation.

    Args:
        pose_info: list [dataset (string), sequence_filepath (string), frame_index (int)]

    Returns:
        pose data, torch tensor of size (1, n_joints*3), all joints considered.
    """

    # load pose data
    assert pose_info[0] in config.supported_datasets, f"Expected data from on of the following datasets: {','.join(config.supported_datasets)} (provided dataset: {pose_info[0]})."
    dp = np.load(os.path.join(config.supported_datasets[pose_info[0]], pose_info[1]))
    # axis angle representation of selected body joints
    pose = dp['poses'][pose_info[2],:].reshape(-1,3) # (n_joints, 3)
    pose = torch.as_tensor(pose).to(dtype=torch.float32)
    # normalize the global orient
    thetax, thetay, thetaz = rotvec_to_eulerangles( pose[:1,:] )
    zeros = torch.zeros_like(thetax)
    pose[0:1,:] = eulerangles_to_rotvec(thetax, thetay, zeros)
    return pose.reshape(1, -1)