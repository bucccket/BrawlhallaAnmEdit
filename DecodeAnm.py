import io
import struct
from typing import Any
import zlib

# TODO implement this in dataclasses
# TODO add type annoations

class ByteReader:
    @staticmethod
    def ReadUint32LE(data: io.BytesIO) -> int:
        byte = data.read(4)
        return int.from_bytes(byte, byteorder="little")

    @staticmethod
    def ReadUint16LE(data: io.BytesIO) -> int:
        byte = data.read(2)
        return int.from_bytes(byte, byteorder="little")

    @staticmethod
    def ReadUint8LE(data: io.BytesIO) -> int:
        byte = data.read(1)
        return int.from_bytes(byte, byteorder="little")

    @staticmethod
    def ReadBoolean(data: io.BytesIO) -> bool:
        return ByteReader.ReadUint8LE(data) != 0

    @staticmethod
    def ReadBytes(data: io.BytesIO, length: int) -> bytes:
        return data.read(length)

    # https://docs.python.org/3/library/struct.html#format-characters
    @staticmethod
    def ReadF32(data: io.BytesIO) -> float:
        byte = data.read(4)
        return struct.unpack('f', byte)[0]

    @staticmethod
    def ReadF64(data: io.BytesIO) -> float:
        byte = data.read(8)
        return struct.unpack('d', byte)[0]


class UTF8String:

    # ctor
    def __init__(self, length: int, string: str):
        self.length = length
        self.string = string

    # static class methods
    @classmethod
    def FromBytesIO(cls, data: io.BytesIO) -> Any:
        length = ByteReader.ReadUint16LE(data)
        res = data.read(length)
        string = res.decode('utf-8')
        return cls(length, string)

    @classmethod
    def FromString(cls, string: str) -> Any:
        return cls(len(string.encode('utf-8')), string)

    # public
    def WriteBytesIO(self, data: io.BytesIO) -> None:
        data.write(self.length.to_bytes(2, byteorder="little"))
        data.write(self.string.encode('utf-8'))

    # overrides
    def __str__(self) -> str:
        return self.string


class AnmBone:
    def __init__(self, data: io.BytesIO):
        self.__ParseFile(data)

    def __ParseFile(self, data: io.BytesIO):
        self.id = ByteReader.ReadUint16LE(data)
        self.opaque = ByteReader.ReadBoolean(data)
        self.matrix: list[list[int]] = []
        if ByteReader.ReadBoolean(data):
            if ByteReader.ReadBoolean(data):
                # Translation matrix
                x = ByteReader.ReadF32(data)
                y = ByteReader.ReadF32(data)
                self.matrix = [[1, 0, x],
                               [0, 1, y],
                               [0, 0, 1]]
            else:
                # Symmetric Matrix
                sc = ByteReader.ReadF32(data)  # scale
                sh = ByteReader.ReadF32(data)  # shear
                x = ByteReader.ReadF32(data)
                y = ByteReader.ReadF32(data)
                self.matrix = [[sc,  sh, x],
                               [sh, -sc, y],
                               [0,    0, 1]]
        else:
            # Full Matrix
            sx = ByteReader.ReadF32(data)  # scale_x
            sh0 = ByteReader.ReadF32(data)  # shear0
            sh1 = ByteReader.ReadF32(data)  # shear1
            sy = ByteReader.ReadF32(data)  # scale_y
            x = ByteReader.ReadF32(data)
            y = ByteReader.ReadF32(data)
            self.matrix = [[sx,  sh1, x],
                           [sh0,  sy, y],
                           [0,     0, 1]]
        self.frame = ByteReader.ReadUint16LE(data)
        self.opacity: float = 1.0
        if self.opaque:
            self.opacity = ByteReader.ReadUint8LE(data) / 255.0


class AnmFrame:
    def __init__(self, data: io.BytesIO):
        self.__ParseFile(data)

    def __ParseFile(self, data: io.BytesIO):
        self.id = ByteReader.ReadUint16LE(data)
        self.offset_a: tuple[int, int] | None = None
        if (ByteReader.ReadBoolean(data)):
            self.offset_a = (
                ByteReader.ReadF64(data),
                ByteReader.ReadF64(data)
            )
        self.offset_b: tuple[int, int] | None = None
        if (ByteReader.ReadBoolean(data)):
            self.offset_b = (
                ByteReader.ReadF64(data),
                ByteReader.ReadF64(data)
            )
        self.rotation = ByteReader.ReadF64(data)
        self.bone_count = ByteReader.ReadUint16LE(data)
        self.bones: dict[int, None | int | AnmBone] = {}
        for i in range(self.bone_count):
            # Huffman Code
            if ByteReader.ReadBoolean(data):
                if ByteReader.ReadBoolean(data):
                    # full copy
                    self.bones[i] = None
                else:
                    # partial copy (update bone anim frame)
                    self.bones[i] = ByteReader.ReadUint16LE(data)
            else:
                self.bones[i] = AnmBone(data)


class AnmAnimation:
    def __init__(self, data: io.BytesIO):
        self.__ParseFile(data)

    def __ParseFile(self, data: io.BytesIO):
        self.name = UTF8String.FromBytesIO(data)
        self.frame_count = ByteReader.ReadUint32LE(data)
        self.loop_start = ByteReader.ReadUint32LE(data)
        self.recovery_start = ByteReader.ReadUint32LE(data)
        self.free_start = ByteReader.ReadUint32LE(data)
        self.preview_frame = ByteReader.ReadUint32LE(data)
        self.base_start = ByteReader.ReadUint32LE(data)
        self.data_size = ByteReader.ReadUint32LE(data)
        self.data_entries = []
        for i in range(self.data_size):
            self.data_entries.append(ByteReader.ReadUint32LE(data))
        self.frame_byte_size = ByteReader.ReadUint32LE(data)
        self.frames = []
        for i in range(self.frame_count):
            self.frames.append(AnmFrame(data))


class AnmClass:
    def __init__(self, data: io.BytesIO):
        self.__ParseFile(data)

    def __ParseFile(self, data: io.BytesIO) -> None:
        self.index: UTF8String = UTF8String.FromBytesIO(data)
        self.filename: UTF8String = UTF8String.FromBytesIO(data)
        self.animationCount: int = ByteReader.ReadUint32LE(data)
        self.animations: list[AnmAnimation] = []
        for i in range(self.animationCount):
            self.animations.append(AnmAnimation(data))


class AnmFile:
    def __init__(self, filename: str):
        fd = open(filename, "rb")

        self.anm_classes: dict[str, AnmClass] = {}

        self.inflated_size = fd.read(4)
        self.zlibdata = fd.read()

        data = io.BytesIO(zlib.decompress(self.zlibdata))

        self.__ParseFile(data)

    def __ParseFile(self, data: io.BytesIO) -> None:
        # open("dump.bin", "wb").write(self.data.getbuffer())
        while (ByteReader.ReadBoolean(data)):
            anmName = UTF8String.FromBytesIO(data)
            anmClass = AnmClass(data)
            self.anm_classes[anmName.string] = anmClass
            exit(0)


if __name__ == "__main__":
    anm_file = AnmFile("Animation_Emote.anm")
