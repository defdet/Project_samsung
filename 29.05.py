def bin_code(str):
    a = []
    for i in range (len(str)):
        a.append(bin(ord(str[i]))[2:])
    return a

def bin_decode(a):
    s = ''
    for i in range (len(a)):
        s += chr(int(('0b'+a[i]),2))
    return s

#тесты:
print(bin_code ('Pososi Pzh'))
print(bin_decode  (['1010000', '1101111', '1110011', '1101111', '1110011', '1101001', '100000', '1010000', '1111010', '1101000']))

