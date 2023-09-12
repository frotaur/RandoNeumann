

def test_confluent(self):
    state = torch.randint(1,9,(1,*size), device=device)
    state = torch.where(torch.rand((1,*size))>0.99,9,state)
    state = torch.zeros_like(state)
    state[0,self.h//2,self.w//2]=1
    state[0,self.h//2,self.w//2+1]=9
    state[0,self.h//2,self.w//2+2]=3
    state[0,self.h//2+1,self.w//2+2]=2
    state[0,self.h//2+1,self.w//2+1]=2
    state[0,self.h//2+1,self.w//2]=4
    state[0,self.h//2-1,self.w//2+1]=3

    self.excitations = torch.randint(0,2,(1,*size), device=device, dtype=torch.uint8)
    self.excitations=torch.zeros_like(self.excitations)
    self.excitations[0,self.h//2,self.w//2]=1