from dataclasses import dataclass
from enum import Enum
import io
import struct
from typing import Any, Generator
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

    @staticmethod
    def ReadF32(data: io.BytesIO) -> float:
        byte = data.read(4)
        return struct.unpack("f", byte)[0]

    @staticmethod
    def ReadF64(data: io.BytesIO) -> float:
        byte = data.read(8)
        return struct.unpack("d", byte)[0]


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
        data.write(struct.pack("f", value))

    @staticmethod
    def WriteF64(data: io.BytesIO, value: float) -> None:
        data.write(struct.pack("d", value))


class UTF8String:
    length: int
    string: str

    # ctor
    def __init__(self, length: int, string: str):
        self.length: int = length
        self.string: str = string

    # static class methods
    @classmethod
    def FromBytesIO(cls, data: io.BytesIO) -> Any:
        length = ByteReader.ReadUint16LE(data)
        res = data.read(length)
        string = res.decode("utf-8")
        return cls(length, string)

    @classmethod
    def FromString(cls, string: str) -> Any:
        return cls(len(string.encode("utf-8")), string)

    # public
    def WriteBytesIO(self, data: io.BytesIO) -> None:
        data.write(self.length.to_bytes(2, byteorder="little"))
        data.write(self.string.encode("utf-8"))

    # overrides
    def __str__(self) -> str:
        return self.string


class Matrix:
    class MatrixType(Enum):
        TRANSLATION = 0
        MIRRORED = 1
        AFFINE = 2

    def __init__(
        self,
        x: float,
        y: float,
        scale_x: float = 1,
        shear0: float = 0,
        shear1: float = 0,
        scale_y: float = 1,
    ):
        self.x: float = x
        self.y: float = y
        self.scale_x: float = scale_x
        self.shear0: float = shear0
        self.shear1: float = shear1
        self.scale_y: float = scale_y

        if scale_x == scale_y == 1 and shear0 == shear1 == 0:
            self.type: self.MatrixType = self.MatrixType.TRANSLATION
        elif scale_x == -scale_y and shear0 == shear1:
            self.type: self.MatrixType = self.MatrixType.MIRRORED
        else:
            self.type: self.MatrixType = self.MatrixType.AFFINE

    def toMatrix(self) -> list[list[float]]:
        return [
            [self.scale_x, self.shear1, self.x],
            [self.shear0, self.scale_y, self.y],
            [0, 0, 1],
        ]

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
            ByteWriter.WriteBoolean(data, self.type == Matrix.MatrixType.TRANSLATION)
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


@dataclass
class AnmBone:
    id: int
    matrix: Matrix
    frame: int
    opacity: float

    @property
    def opaque(self) -> bool:
        return self.opacity == 1.0

    @classmethod
    def FromBytesIO(cls, data: io.BytesIO) -> None:
        id: int = ByteReader.ReadUint16LE(data)
        opaque: bool = ByteReader.ReadBoolean(data)
        matrix: Matrix = Matrix.FromBytes(data)
        frame: int = ByteReader.ReadUint16LE(data)
        opacity: float = 1.0 if opaque else ByteReader.ReadUint8LE(data) / 255.0
        return cls(id, matrix, frame, opacity)

    def WriteBytesIO(self, data: io.BytesIO) -> None:
        ByteWriter.WriteUint16LE(data, self.id)
        ByteWriter.WriteBoolean(data, self.opaque)
        self.matrix.WriteBytesIO(data)
        ByteWriter.WriteUint16LE(data, self.frame)
        if not self.opaque:
            ByteWriter.WriteUint8LE(data, int(self.opacity * 255.0))


@dataclass
class AnmFrame:

    id: int
    offset_a: tuple[int, int] | None
    offset_b: tuple[int, int] | None
    rotation: float
    bones: list[None | int | AnmBone]

    @classmethod
    def FromBytesIO(cls, data: io.BytesIO) -> None:
        id: int = ByteReader.ReadUint16LE(data)
        offset_a: tuple[int, int] | None = (
            (ByteReader.ReadF64(data), ByteReader.ReadF64(data))
            if ByteReader.ReadBoolean(data)
            else None
        )
        offset_b: tuple[int, int] | None = (
            (ByteReader.ReadF64(data), ByteReader.ReadF64(data))
            if ByteReader.ReadBoolean(data)
            else None
        )
        rotation: float = ByteReader.ReadF64(data)
        bone_count: int = ByteReader.ReadUint16LE(data)
        bones: list[None | int | AnmBone] = []
        for _ in range(bone_count):
            # Huffman Code
            if ByteReader.ReadBoolean(data):
                if ByteReader.ReadBoolean(data):
                    # full copy
                    bones.append(None)
                else:
                    # partial copy (update bone anim frame)
                    bones.append(ByteReader.ReadUint16LE(data))
            else:
                bones.append(AnmBone.FromBytesIO(data))
        return cls(id, offset_a, offset_b, rotation, bones)

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
        ByteWriter.WriteUint16LE(data, len(self.bones))
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


@dataclass
class AnmAnimation:
    name: UTF8String
    loop_start: int
    recovery_start: int
    free_start: int
    preview_frame: int
    base_start: int
    data_entries: list[int]
    frames: Generator[AnmFrame, None, None]

    @classmethod
    def FromBytesIO(cls, data: io.BytesIO) -> None:
        name: UTF8String = UTF8String.FromBytesIO(data)
        frame_count: int = ByteReader.ReadUint32LE(data)
        loop_start: int = ByteReader.ReadUint32LE(data)
        recovery_start: int = ByteReader.ReadUint32LE(data)
        free_start: int = ByteReader.ReadUint32LE(data)
        preview_frame: int = ByteReader.ReadUint32LE(data)
        base_start: int = ByteReader.ReadUint32LE(data)
        data_size: int = ByteReader.ReadUint32LE(data)
        data_entries: list[int] = [
            ByteReader.ReadUint32LE(data) for _ in range(data_size)
        ]
        frame_byte_size: int = ByteReader.ReadUint32LE(data)
        sub_buffer = io.BytesIO(data.read(frame_byte_size))
        frames: Generator[AnmFrame, None, None] = (
            AnmFrame.FromBytesIO(sub_buffer) for _ in range(frame_count)
        )

        return cls(
            name,
            loop_start,
            recovery_start,
            free_start,
            preview_frame,
            base_start,
            data_entries,
            frames,
        )

    def WriteBytesIO(self, data: io.BytesIO) -> None:
        self.name.WriteBytesIO(data)
        frames = list(self.frames)
        ByteWriter.WriteUint32LE(data, len(frames))
        ByteWriter.WriteUint32LE(data, self.loop_start)
        ByteWriter.WriteUint32LE(data, self.recovery_start)
        ByteWriter.WriteUint32LE(data, self.free_start)
        ByteWriter.WriteUint32LE(data, self.preview_frame)
        ByteWriter.WriteUint32LE(data, self.base_start)
        ByteWriter.WriteUint32LE(data, len(self.data_entries))
        for data_entry in self.data_entries:
            ByteWriter.WriteUint32LE(data, data_entry)
        frame_data = io.BytesIO()
        for frame in frames:
            frame.WriteBytesIO(frame_data)
        # TODO calculate frame_byte_size
        ByteWriter.WriteUint32LE(data, frame_data.getbuffer().nbytes)
        ByteWriter.WriteBytes(data, frame_data.getbuffer())


@dataclass
class AnmClass:
    index: UTF8String
    filename: UTF8String
    animations: list[AnmAnimation]

    @classmethod
    def FromBytesIO(cls, data: io.BytesIO) -> None:
        index: UTF8String = UTF8String.FromBytesIO(data)
        filename: UTF8String = UTF8String.FromBytesIO(data)
        animationCount: int = ByteReader.ReadUint32LE(data)
        animations: list[AnmAnimation] = [
            AnmAnimation.FromBytesIO(data) for _ in range(animationCount)
        ]
        return cls(index, filename, animations)

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
        while ByteReader.ReadBoolean(data):
            anmName: UTF8String = UTF8String.FromBytesIO(data)
            anmClass: AnmClass = AnmClass.FromBytesIO(data)
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
