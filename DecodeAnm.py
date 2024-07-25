from enum import Enum
import io
import struct
from typing import Any
import zlib

# TODO implement this in dataclasses


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


class ByteWriter:
    @staticmethod
    def WriteUint32LE(data: io.BytesIO, value: int) -> None:
        data.write(value.to_bytes(4, byteorder="little"))

    @staticmethod
    def WriteUint16LE(data: io.BytesIO, value: int) -> None:
        data.write(value.to_bytes(2, byteorder="little"))

    @staticmethod
    def WriteUint8LE(data: io.BytesIO, value: int) -> None:
        data.write(value.to_bytes(1, byteorder="little"))

    @staticmethod
    def WriteBoolean(data: io.BytesIO, value: bool) -> None:
        ByteWriter.WriteUint8LE(data, 1 if value else 0)

    @staticmethod
    def WriteBytes(data: io.BytesIO, value: bytes) -> None:
        data.write(value)

    @staticmethod
    def WriteF32(data: io.BytesIO, value: float) -> None:
        data.write(struct.pack('f', value))

    @staticmethod
    def WriteF64(data: io.BytesIO, value: float) -> None:
        data.write(struct.pack('d', value))


class UTF8String:

    # ctor
    def __init__(self, length: int, string: str):
        self.length: int = length
        self.string: str = string

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


class Matrix:
    class MatrixType(Enum):
        TRANSLATION = 0
        MIRRORED = 1
        AFFINE = 2

    def __init__(self, x: float, y: float, scale_x: float = 1, shear0: float = 0, shear1: float = 0, scale_y: float = 1):
        self.x: float = x
        self.y: float = y
        self.scale_x: float = scale_x
        self.shear0: float = shear0
        self.shear1: float = shear1
        self.scale_y: float = scale_y

        if scale_x == scale_y == 1 and shear0 == shear1 == 0:
            self.type: Matrix.MatrixType = Matrix.MatrixType.TRANSLATION
        elif scale_x == -scale_y and shear0 == shear1:
            self.type: Matrix.MatrixType = Matrix.MatrixType.MIRRORED
        else:
            self.type: Matrix.MatrixType = Matrix.MatrixType.AFFINE

    def toMatrix(self) -> list[list[float]]:
        return [[self.scale_x,  self.shear1, self.x],
                [self.shear0,  self.scale_y, self.y],
                [0,                       0,      1]]

    @classmethod
    def FromBytes(cls, data: io.BytesIO) -> Any:
        if ByteReader.ReadBoolean(data):
            if ByteReader.ReadBoolean(data):
                # Translation matrix
                x = ByteReader.ReadF32(data)
                y = ByteReader.ReadF32(data)
                return cls(x, y)
            else:
                # Symmetric Matrix
                sc = ByteReader.ReadF32(data)  # scale
                sh = ByteReader.ReadF32(data)  # shear
                x = ByteReader.ReadF32(data)
                y = ByteReader.ReadF32(data)
                return cls(x, y, sc, sh, sh, -sc)
        else:
            # Full Matrix
            sx = ByteReader.ReadF32(data)  # scale_x
            sh0 = ByteReader.ReadF32(data)  # shear0
            sh1 = ByteReader.ReadF32(data)  # shear1
            sy = ByteReader.ReadF32(data)  # scale_y
            x = ByteReader.ReadF32(data)
            y = ByteReader.ReadF32(data)
            return cls(x, y, sx, sh0, sh1, sy)

    def WriteBytesIO(self, data: io.BytesIO) -> None:
        ByteWriter.WriteBoolean(data, self.type != Matrix.MatrixType.AFFINE)
        if self.type != Matrix.MatrixType.AFFINE:
            ByteWriter.WriteBoolean(
                data, self.type == Matrix.MatrixType.TRANSLATION)
            if self.type == Matrix.MatrixType.MIRRORED:
                ByteWriter.WriteF32(data, self.scale_x)
                ByteWriter.WriteF32(data, self.shear0)
        else:
            ByteWriter.WriteF32(data, self.scale_x)
            ByteWriter.WriteF32(data, self.shear0)
            ByteWriter.WriteF32(data, self.shear1)
            ByteWriter.WriteF32(data, self.scale_y)
        ByteWriter.WriteF32(data, self.x)
        ByteWriter.WriteF32(data, self.y)


class AnmBone:
    def __init__(self, data: io.BytesIO):
        self.__ParseFile(data)

    def __ParseFile(self, data: io.BytesIO) -> None:
        self.id: int = ByteReader.ReadUint16LE(data)
        self.opaque: bool = ByteReader.ReadBoolean(data)
        self.matrix: Matrix = Matrix.FromBytes(data)
        self.frame: int = ByteReader.ReadUint16LE(data)
        self.opacity: float = 1.0
        if not self.opaque:
            self.opacity = ByteReader.ReadUint8LE(data) / 255.0

    def WriteBytesIO(self, data: io.BytesIO) -> None:
        ByteWriter.WriteUint16LE(data, self.id)
        ByteWriter.WriteBoolean(data, self.opaque)
        self.matrix.WriteBytesIO(data)
        ByteWriter.WriteUint16LE(data, self.frame)
        if not self.opaque:
            ByteWriter.WriteUint8LE(data, int(self.opacity * 255))


class AnmFrame:
    def __init__(self, data: io.BytesIO):
        self.__ParseFile(data)

    def __ParseFile(self, data: io.BytesIO) -> None:
        self.id: int = ByteReader.ReadUint16LE(data)
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
        self.rotation: float = ByteReader.ReadF64(data)
        self.bone_count: int = ByteReader.ReadUint16LE(data)
        self.bones: list[None | int | AnmBone] = []
        for i in range(self.bone_count):
            # Huffman Code
            if ByteReader.ReadBoolean(data):
                if ByteReader.ReadBoolean(data):
                    # full copy
                    self.bones.append(None)
                else:
                    # partial copy (update bone anim frame)
                    self.bones.append(ByteReader.ReadUint16LE(data))
            else:
                self.bones.append(AnmBone(data))

    def WriteBytesIO(self, data: io.BytesIO) -> None:
        ByteWriter.WriteUint16LE(data, self.id)
        ByteWriter.WriteBoolean(data, self.offset_a is not None)
        if self.offset_a is not None:
            ByteWriter.WriteF64(data, self.offset_a[0])
            ByteWriter.WriteF64(data, self.offset_a[1])
        ByteWriter.WriteBoolean(data, self.offset_b is not None)
        if self.offset_b is not None:
            ByteWriter.WriteF64(data, self.offset_b[0])
            ByteWriter.WriteF64(data, self.offset_b[1])
        ByteWriter.WriteF64(data, self.rotation)
        ByteWriter.WriteUint16LE(data, self.bone_count)
        for bone in self.bones:
            if bone is None:
                ByteWriter.WriteBoolean(data, True)
                ByteWriter.WriteBoolean(data, True)
            elif isinstance(bone, int):
                ByteWriter.WriteBoolean(data, True)
                ByteWriter.WriteBoolean(data, False)
                ByteWriter.WriteUint16LE(data, bone)
            else:
                ByteWriter.WriteBoolean(data, False)
                bone.WriteBytesIO(data)


class AnmAnimation:
    def __init__(self, data: io.BytesIO):
        self.__ParseFile(data)

    def __ParseFile(self, data: io.BytesIO) -> None:
        self.name: UTF8String = UTF8String.FromBytesIO(data)
        self.frame_count: int = ByteReader.ReadUint32LE(data)
        self.loop_start: int = ByteReader.ReadUint32LE(data)
        self.recovery_start: int = ByteReader.ReadUint32LE(data)
        self.free_start: int = ByteReader.ReadUint32LE(data)
        self.preview_frame: int = ByteReader.ReadUint32LE(data)
        self.base_start: int = ByteReader.ReadUint32LE(data)
        self.data_size: int = ByteReader.ReadUint32LE(data)
        self.data_entries: list[int] = []
        for i in range(self.data_size):
            self.data_entries.append(ByteReader.ReadUint32LE(data))
        self.frame_byte_size: int = ByteReader.ReadUint32LE(data)
        self.frames: list[AnmFrame] = []
        for i in range(self.frame_count):
            self.frames.append(AnmFrame(data))

    def WriteBytesIO(self, data: io.BytesIO) -> None:
        self.name.WriteBytesIO(data)
        ByteWriter.WriteUint32LE(data, len(self.frames))
        ByteWriter.WriteUint32LE(data, self.loop_start)
        ByteWriter.WriteUint32LE(data, self.recovery_start)
        ByteWriter.WriteUint32LE(data, self.free_start)
        ByteWriter.WriteUint32LE(data, self.preview_frame)
        ByteWriter.WriteUint32LE(data, self.base_start)
        ByteWriter.WriteUint32LE(data, len(self.data_entries))
        for data_entry in self.data_entries:
            ByteWriter.WriteUint32LE(data, data_entry)
        frame_data = io.BytesIO()
        for frame in self.frames:
            frame.WriteBytesIO(frame_data)
        # TODO calculate frame_byte_size
        ByteWriter.WriteUint32LE(data, frame_data.getbuffer().nbytes)
        ByteWriter.WriteBytes(data, frame_data.getbuffer())


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

    def WriteBytesIO(self, data: io.BytesIO) -> None:
        self.index.WriteBytesIO(data)
        self.filename.WriteBytesIO(data)
        ByteWriter.WriteUint32LE(data, len(self.animations))
        for animation in self.animations:
            animation.WriteBytesIO(data)


class AnmFile:
    def __init__(self, filename: str):
        try:
            with open(filename, "rb") as fd:
                self.anm_classes: dict[str, AnmClass] = {}

                inflated_size = fd.read(4)
                zlibdata = fd.read()

                data = io.BytesIO(zlib.decompress(zlibdata))

                self.__ParseFile(data)
        except FileNotFoundError:
            print(f"File {filename} not found")
            return

    def __ParseFile(self, data: io.BytesIO) -> None:
        # open("dump.bin", "wb").write(data.getbuffer())
        while (ByteReader.ReadBoolean(data)):
            anmName: UTF8String = UTF8String.FromBytesIO(data)
            anmClass: AnmClass = AnmClass(data)
            self.anm_classes[anmName.string] = anmClass

    def Save(self, filename: str) -> None:
        data = io.BytesIO()
        self.WriteBytesIO(data)
        inflated_size = data.getbuffer().nbytes.to_bytes(4, byteorder="little")
        zlibdata = zlib.compress(data.getbuffer())
        with open(filename, "wb") as fd:
            fd.write(inflated_size)
            fd.write(zlibdata)

    def WriteBytesIO(self, data: io.BytesIO) -> None:
        for anmName, anmClass in self.anm_classes.items():
            ByteWriter.WriteBoolean(data, True)
            UTF8String.FromString(anmName).WriteBytesIO(data)
            anmClass.WriteBytesIO(data)
        ByteWriter.WriteBoolean(data, False)
