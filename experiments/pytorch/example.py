import torch

src = list(range(9))
a = torch.Tensor(src)
print(a)

b = a.view(3, 3) # changes the shape of a
print(b)

c = b[1:, 1:]
print(c)
print(c.shape,
    "\nStride:", c.stride(),
    "\nOffset:", c.storage_offset())

d = torch.sqrt(c)
print(d)

print("c'", c.sqrt_())