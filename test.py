import torch
import torch.nn.functional as F

print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("Cuda is Availabe")
else:
    print("Cuda Can't be found")      

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

t = torch.tensor([1,2], device=device)  
print(t)

a = torch.randn(4, 6)
b = torch.randint(6, (4,), dtype=torch.int64)
loss = F.cross_entropy(a, b)
print(loss)



device = "cpu"

a = torch.Tensor([[-10.3353, -28.4371,   2.0768,   -4.2789,  -8.6644,  -6.0815],
        [-10.3353, -28.4371,   2.0768,   -4.2789,  -8.6644,  -6.0815],
        [-10.3353, -28.4371,   2.0768,   -4.2789,  -8.6644,  -6.0815],
        [-10.3353, -28.4371,   2.0768,   -4.2789,  -8.6644,  -6.0815]]).to(device)
b = torch.Tensor([ -100,  -1,  -100,  2456]).long().to(device)

loss = torch.nn.functional.cross_entropy(a,b,ignore_index=-1)
print(loss)
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118