#include <cstdint>

uint16_t PY_CRC_16_T8_USB_i(uint8_t *di, uint32_t len)
{
	uint16_t crc_poly = 0xA001; //Bit sequence inversion of 0x8005
	uint16_t data_t = 0xFFFF; //CRC register

    for(uint32_t i = 0; i < len; i++)
    {
    	data_t ^= di[i]; //8-bit data

        for (uint8_t j = 0; j < 8; j++)
        {
            if (data_t & 0x0001)
            	data_t = (data_t >> 1) ^ crc_poly;
            else
            	data_t >>= 1;
        }
    }

    return data_t ^ 0xFFFF;
}