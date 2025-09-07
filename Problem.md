Problem (unicode1)

a) '\x00'

b)  (\__repr\_\_()) will show clearly what the string looks like, but (\_\_str\_\_()) shows user-friendly without any invisible character.

c) "this is a test" + chr(0) + "string" -> 'this is a test\x00string' 

print("this is a test" + chr(0) + "string") -> this is a teststring

Problem (unicode2)

a) utf-8 is flexible, since it can use at least 1 byte to present a simple character, while utf-16 will at least use 16 digits(2 bytes) to present a character. Also, utf-32 use 4 bytes. Both utf-16 and 32 cost more than utf-8.

b) Because it will decode a multi-byte character into many single-byte characters which are undecodable -- for avoiding repeating and ambiguity. For example, '牛'

c) '牛' is decoded as b'\xe7\x89\x9b', so if we use first 2 bytes, b'\xe7\x89', the decoder can not recognize them.

