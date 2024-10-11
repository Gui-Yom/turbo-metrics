struct IvfHeader {
    u32 sig;
    u16 version;
    u16 length;
    char fourcc[4];
    u16 width;
    u16 height;
    u32 den;
    u32 num;
    u32 frames;
    u32 unused;
};

struct IvfPacket {
    u32 len;
    u64 pts;
    u8 data[len];
};

IvfHeader header @ 0x0;
IvfPacket frames[header.frames] @ $;