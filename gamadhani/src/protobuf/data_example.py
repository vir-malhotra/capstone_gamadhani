from typing import Literal, Optional

import numpy as np
import torch
import pdb

from gamadhani.src.protobuf.data_pb2 import DataVal as DataPB

DTYPE_TO_PRECISION = {
    np.int16: DataPB.Precision.INT16,
    np.int32: DataPB.Precision.INT32,
    np.int64: DataPB.Precision.INT64,
    np.float16: DataPB.Precision.FLOAT16,
    np.float32: DataPB.Precision.FLOAT32,
    np.float64: DataPB.Precision.FLOAT64,
}

PRECISION_TO_DTYPE = {
    DataPB.Precision.INT16: np.int16,
    DataPB.Precision.INT32: np.int32,
    DataPB.Precision.INT64: np.int64,
    DataPB.Precision.FLOAT16: np.float16,
    DataPB.Precision.FLOAT32: np.float32,
    DataPB.Precision.FLOAT64: np.float64,
}


class AudioExample(object):

    def __init__(
            self,
            b: Optional[str] = None,
            output_type: Literal["numpy", "torch"] = "numpy") -> None:
        if b is not None:
            self.ae = DataPB.FromString(b)
        else:
            self.ae = DataPB()

        self.output_type = output_type

    def get(self, keys: list):
        for key in keys:
            buf = self.ae.buffers[key]
            if buf is None:
                raise KeyError(f"key '{key}' not available")

            array = np.frombuffer(
                buf.data,
                dtype=PRECISION_TO_DTYPE[buf.precision],
            ).reshape(buf.shape).copy()

            if self.output_type == "numpy":
                pass
            elif self.output_type == "torch":
                array = torch.from_numpy(array)
            else:
                raise ValueError(f"Output type {self.output_type} not available")
            return {
                'data': array,
                'sampling_rate': buf.sampling_rate,
                'data_path': buf.data_path,
                'start_time': buf.start_time,
            }

    def put(self, arrays: dict, dtype: np.dtype, sample_rate: Optional[int], data_path: Optional[str], start_time: Optional[float], global_conditions: Optional[dict] = None):
        # pdb.set_trace()
        for key, array in arrays.items():
            buffer = self.ae.buffers[key]
            buffer.data = np.asarray(array).astype(dtype).tobytes()
            buffer.shape.extend(array.shape)
            buffer.precision = DTYPE_TO_PRECISION[dtype]
            buffer.sampling_rate = sample_rate
            buffer.data_path = data_path
            buffer.start_time = start_time
        if global_conditions is not None:
            global_conditions.tonic = global_conditions['tonic']
            global_conditions.raga = global_conditions['raga']
            global_conditions.singer = global_conditions['singer']


    def as_dict(self):
        vals = {k: self.get([k]) for k in self.ae.buffers}
        vals['global_conditions'] = {
            'tonic': self.ae.global_conditions.tonic,
            'raga': self.ae.global_conditions.raga,
            'singer': self.ae.global_conditions.singer,
        }
        return vals

    def __str__(self) -> str:
        repr = []
        repr.append("DataVal(")
        for key in self.ae.buffers:
            array = self.get(key)
            repr.append(f"\t{key}[{array.dtype}] {array.shape},")
        repr.append(")")
        return "\n".join(repr)

    def __bytes__(self) -> str:
        return self.ae.SerializeToString()
