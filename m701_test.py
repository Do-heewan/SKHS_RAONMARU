import serial

PORT = "/dev/ttyAMA0"
BAUD = 9600

ser = serial.Serial(
    port=PORT,
    baudrate=BAUD,
    bytesize=8,
    parity=serial.PARITY_NONE,
    stopbits=1,
    timeout=2
)
print("M701 UART Test")
print("Ready for data receiving...")
print("Ctrl+C is Shutdown")

while True:
    data = ser.read(17)

    if len(data) == 17:
        print("Receiving :", data.hex(" "))

        if data[0] == 0x3C and data[1] == 0x02:
            print("  ★ M701 Data Frame is good!")
        else:
            print("  Frame start bytes are not correct")
