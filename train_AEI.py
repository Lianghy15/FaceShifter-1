from network.AEI_Net import *
from network.MultiscaleDiscriminator import *
from utils.Dataset import FaceEmbed, With_Identity
from torch.utils.data import DataLoader
import torch.optim as optim
from face_modules.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch.nn.functional as F
import torch
import time
import torchvision
import cv2
from apex import amp
import visdom
from same_identity_dataset import *
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb 
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='debug', required=True)
parser.add_argument('-g', '--gpu', type=str, default='cuda:0', required=True)
args = parser.parse_args()

if not os.path.exists('./log/' + args.name):
    os.mkdir('./log/' + args.name)
writer = SummaryWriter('./log/'+ args.name)
batch_size = 10
val_batch_size = 1
lr_G = 4e-4
lr_D = 4e-4
max_epoch = 30
show_step = 1
save_epoch = 1
model_save_path = './saved_models/' + args.name
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
optim_level = 'O1'

# fine_tune_with_identity = False

device = torch.device(args.gpu)
# torch.set_num_threads(12)

G = AEI_Net(c_id=512).to(device)
D = MultiscaleDiscriminator(input_nc=3, n_layers=6, norm_layer=torch.nn.InstanceNorm2d).to(device)
G.train()
D.train()

arcface = Backbone(50, 0.6, 'ir_se').to(device)
arcface.eval()
arcface.load_state_dict(torch.load('./face_modules/model_ir_se50.pth', map_location=device), strict=False)

opt_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0, 0.999))

G, opt_G = amp.initialize(G, opt_G, opt_level=optim_level)
D, opt_D = amp.initialize(D, opt_D, opt_level=optim_level)

try:
    G.load_state_dict(torch.load('saved_models/G_latest.pth', map_location=torch.device('cpu')), strict=False)
    D.load_state_dict(torch.load(model_save_path + '/D_latest.pth', map_location=torch.device('cpu')), strict=False)
except Exception as e:
    print(e)

# if not fine_tune_with_identity:
    # dataset = FaceEmbed(['../celeb-aligned-256_0.85/', '../ffhq_256_0.85/', '../vgg_256_0.85/', '../stars_256_0.85/'], same_prob=0.5)
# else:
    # dataset = With_Identity('../washed_img/', 0.8)
#dataset = FaceEmbed(['preprocess'], same_prob=0.8)
train_path = '../data/train_data'
val_path = '../data/val_data'

transform = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
#transform = transforms.Compose([
#            transforms.Resize((256, 256)),
#            transforms.CenterCrop((256, 256)),
#            transforms.ToTensor(),
#            ])
dataset_train = AEI_Dataset_identity(train_path, transform=transform)
dataset_val = AEI_Val_Dataset_identity(val_path, transform=transform)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(dataset_val, batch_size=val_batch_size, shuffle=False, num_workers=0, drop_last=False)


MSE = torch.nn.MSELoss()
L1 = torch.nn.L1Loss()


def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1-X).mean()
    else:
        return torch.relu(X+1).mean()


def get_grid_image(X):
    X = X[:8]
    X = torchvision.utils.make_grid(X.detach().cpu(), nrow=X.shape[0]) * 0.5 + 0.5
    return X


def make_image(Xs, Xt, Y):
    Xs = get_grid_image(Xs)
    Xt = get_grid_image(Xt)
    Y = get_grid_image(Y)
    return torch.cat((Xs, Xt, Y), dim=1).numpy()


# prior = torch.FloatTensor(cv2.imread('./prior.png', 0).astype(np.float)/255).to(device)

print(torch.backends.cudnn.benchmark)
#torch.backends.cudnn.benchmark = True
def validation(G, D, val_loader, epoch, val_batch_size):
    G.eval()
    D.eval()
    out = []
    for idx, batch in enumerate(val_loader):
        source_img, target_img, same = batch
        source_img, target_img, same = source_img.to(device), target_img.to(device), same.to(device)
        embed, Xs_feats = arcface(F.interpolate(source_img[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
        Y, Xt_attr = G(target_img, embed)

        Di = D(Y)
        L_adv = 0

        for di in Di:
            L_adv += hinge_loss(di[0], True)
        

        Y_aligned = Y[:, :, 19:237, 19:237]
        ZY, Y_feats = arcface(F.interpolate(Y_aligned, [112, 112], mode='bilinear', align_corners=True))
        L_id =(1 - torch.cosine_similarity(embed, ZY, dim=1)).mean()

        Y_attr = G.get_attr(Y)
        L_attr = 0
        for i in range(len(Xt_attr)):
            L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(val_batch_size, -1), dim=1).mean()
        L_attr /= 2.0

        L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - target_img, 2).reshape(val_batch_size, -1), dim=1) * same) / (same.sum() + 1e-6)

        lossG = 1*L_adv.item() + 10*L_attr.item() + 10*L_id.item() + 1*L_rec.item()
        #lossG = 1*L_adv.item() + 10*L_attr.item() + 10*L_id.item() 
        image = torch.tensor(make_image(source_img, target_img, Y))
        out += [image.cpu()]
    out = torchvision.utils.make_grid(out, nrow=len(val_loader)).numpy()
#    print(out.shape)
#    print(np.min(out), np.max(out))
    return out, lossG
    

VAL_LOSS = 100000       
        
for epoch in range(0, max_epoch):
#************************for debug*************************************
#    out, val_loss = validation(G, D, val_loader, epoch, val_batch_size)
#    print('************************')
#    if not os.path.exists('gen_images/' + args.name):
#        os.mkdir('gen_images/' + args.name)
#    cv2.imwrite('./gen_images/' + args.name +'/epoch={}.jpg'.format(epoch), out.transpose(1, 2, 0)*255)
#    writer.add_image('gen_images', out, global_step=epoch)
#***********************************************************************
    # torch.cuda.empty_cache()
    G.train()
    D.train()
    loop = tqdm(train_loader, miniters=100)
    count = 0
    for iteration, data in enumerate(loop):
        start_time = time.time()
        Xs, Xt, same_person = data
        cur = same_person.squeeze().sum().item()
        count += cur
        Xs = Xs.to(device)
        Xt = Xt.to(device)
        # embed = embed.to(device)
        with torch.no_grad():
            embed, Xs_feats = arcface(F.interpolate(Xs[:, :, 19:237, 19:237], [112, 112], mode='bilinear', align_corners=True))
        same_person = same_person.to(device)
        #diff_person = (1 - same_person)

        # train G
        opt_G.zero_grad()
        Y, Xt_attr = G(Xt, embed)

        Di = D(Y)
        L_adv = 0

        for di in Di:
            L_adv += hinge_loss(di[0], True)
        

        Y_aligned = Y[:, :, 19:237, 19:237]
        ZY, Y_feats = arcface(F.interpolate(Y_aligned, [112, 112], mode='bilinear', align_corners=True))
        L_id =(1 - torch.cosine_similarity(embed, ZY, dim=1)).mean()

        Y_attr = G.get_attr(Y)
        L_attr = 0
        for i in range(len(Xt_attr)):
            L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(batch_size, -1), dim=1).mean()
        L_attr /= 2.0

        L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

        #lossG = 1*L_adv + 10*L_attr + 5*L_id 
        lossG = 1*L_adv + 10*L_attr + 5*L_id + 1*L_rec
        with amp.scale_loss(lossG, opt_G) as scaled_loss:
            scaled_loss.backward()

        # lossG.backward()
        opt_G.step()

        # train D
        opt_D.zero_grad()
        # with torch.no_grad():
        #     Y, _ = G(Xt, embed)
        fake_D = D(Y.detach())
        loss_fake = 0
        for di in fake_D:
            loss_fake += hinge_loss(di[0], False)

        true_D = D(Xs)
        loss_true = 0
        for di in true_D:
            loss_true += hinge_loss(di[0], True)
        # true_score2 = D(Xt)[-1][0]

        lossD = 0.5*(loss_true.mean() + loss_fake.mean())

        with amp.scale_loss(lossD, opt_D) as scaled_loss:
            scaled_loss.backward()
        # lossD.backward()
        opt_D.step()
        batch_time = time.time() - start_time
   
        loop.set_description(f"Epoch [{epoch}] batch [{iteration}/{len(train_loader)}]")
        loop.set_postfix(loss=lossG.item())
     
        writer.add_scalar('lossD', lossD.item(), iteration)
        writer.add_scalar('lossG', lossG.item(), iteration)
        writer.add_scalar('L_adv', L_adv.item(), iteration)
        writer.add_scalar('L_id', L_id.item(), iteration)
        writer.add_scalar('L_attr', L_attr.item(), iteration)
        writer.add_scalar('L_rec', L_rec.item(), iteration)
    if epoch % show_step == 0:
        out, val_loss = validation(G, D, val_loader, epoch, val_batch_size)
        cv2.imwrite('./gen_images/' + args.name + '/epoch={}.jpg'.format(epoch), out*255)

        writer.add_image('gen_images', out, global_step=epoch)
        writer.add_scalar('val_loss', val_loss, global_step=epoch)
    if val_loss < VAL_LOSS:
        VAL_LOSS = val_loss
        torch.save(G.state_dict(), model_save_path + '/G_latest_epoch{}.pth'.format(epoch))
        torch.save(D.state_dict(), model_save_path + '/D_latest_epoch{}.pth'.format(epoch))   
    writer.add_scalar('same_identity_ratio', count/(batch_size*len(train_loader)), global_step=epoch)


