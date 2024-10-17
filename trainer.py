from load_data import get_all_dataloaders
from models_clip_based import *
from ops import *
from utils.calc_hammingranking import calc_map
import os
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from scipy.io import savemat


class TrainerV0:
    def __init__(self, cfg):
        self.cfg = cfg

        # include hyper parameters
        self.alpha = 1.0
        self.eta = 100
        self.gamma = 1.0
        self.margin = 1.0

        self.loaders, self.orig_data = get_all_dataloaders(cfg)
        self.checkpoint_path = cfg.checkpoint_path

        self.Epoch = cfg.Epoch
        self.lr_lab = cfg.lr_lab
        self.lr_img = cfg.lr_img
        self.lr_txt = cfg.lr_txt
        self.k_lab_net = cfg.k_lab_net
        self.k_img_net = cfg.k_img_net
        self.k_txt_net = cfg.k_txt_net
        self.lnet = LabelNet(cfg).cuda()
        self.lnet.train()
        self.model = FMUN(cfg).cuda()
        self.model.train()
        self.lnet_opt = torch.optim.Adam(self.lnet.parameters(), lr=self.lr_lab[0])
        self.inet_opt = torch.optim.Adam(self.model.inet.parameters(), lr=self.lr_img[0])
        self.tnet_opt = torch.optim.Adam(self.model.tnet.parameters(), lr=self.lr_txt[0])

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.num_train = self.orig_data['L']['train'].shape[0]
        self.gt_ratio = cfg.known_ratio

        self.bit = cfg.bit
        self.SEMANTIC_EMBED = cfg.SEMANTIC_EMBED
        self.batch_size = cfg.batch_size


    def train(self):
        var = {}
        var['v'] = np.clip(np.random.randn(self.num_train, self.bit).astype(np.float32), -1, 1)
        var['t'] = np.clip(np.random.randn(self.num_train, self.bit).astype(np.float32), -1, 1)
        # var['l'] = np.clip(np.random.randn(self.num_train, self.bit).astype(np.float32), -1, 1)

        var['vf'] = np.random.randn(self.num_train, self.SEMANTIC_EMBED).astype(np.float32)
        var['tf'] = np.random.randn(self.num_train, self.SEMANTIC_EMBED).astype(np.float32)
        var['B'] = np.sign(var['v'] + var['t'])
        # loss_l_list = []
        condition_dir = './result-bit-%d-ratio-%f' % (self.cfg.bit, self.gt_ratio)

        if not os.path.exists(condition_dir + '/Ls.mat'):
            if os.path.exists('./checkpoint/prompt_net.ckp'):
                print('Loading prompt_net from checkpoint.')
                lnet_state = torch.load('./checkpoint/prompt_net.ckp')['prompt_net']
                self.lnet.load_state_dict(lnet_state)
            else:
                for k in range(30):
                    ## train lnet
                    print(f'++++++++ Training prompt_net epoch {k} ++++++++')
                    lnet_loss = self.train_lab_net(var)
                    # loss_l_list.append(lnet_loss)
                    print('prompt_net loss_total: %f' % lnet_loss)
                    # save checkpoint
                    state = {
                        'prompt_net': self.lnet.state_dict(),
                    }
                    torch.save(state, './checkpoint/prompt_net.ckp')
            print('============================== both ==============================')
            # L_ic_new = self.clip_recovery_from_present_one_step('both', margin=1.0)
            L_ic_new = self.clip_recovery_from_present('both')
            self.orig_data['L']['train'][:, :] = L_ic_new[:, :]
            if not os.path.exists(condition_dir):
                os.mkdir(condition_dir)
            savemat(os.path.join(condition_dir, 'Ls.mat'), {
                'L_ic': self.orig_data['L']['ic_train'],
                'L_rec': self.orig_data['L']['train'],
                'index_all': self.orig_data['index_all']
            })

        # Iterations
        for epoch in range(self.Epoch):
            self.model.train()
            # adjust lr
            lr_inet = self.lr_img[epoch]
            lr_tnet = self.lr_txt[epoch]
            adjust_learning_rate(self.inet_opt, lr_inet)
            adjust_learning_rate(self.tnet_opt, lr_tnet)
            var['epoch'] = epoch
            train_model_loss = self.train_both_net(var)
            print('model_net loss_total: %d' % train_model_loss)
            var['B'] = np.sign(var['v'] + var['t'])

            self.model.eval()
            # evaluation after some epoch
            with torch.no_grad():
                qBX, qBY = self.generate_code(self.loaders['qloader'])
                rBX, rBY = self.generate_code(self.loaders['rloader'])
                mapi2t = calc_map(qBX, rBY, self.orig_data['L']['query'], self.orig_data['L']['retrieval'])
                mapt2i = calc_map(qBY, rBX, self.orig_data['L']['query'], self.orig_data['L']['retrieval'])
                mapi2i = calc_map(qBX, rBX, self.orig_data['L']['query'], self.orig_data['L']['retrieval'])
                mapt2t = calc_map(qBY, rBY, self.orig_data['L']['query'], self.orig_data['L']['retrieval'])

                if not os.path.exists(condition_dir):
                    os.mkdir(condition_dir)

                save_dir_name = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
                cur_dir_path = os.path.join(condition_dir, save_dir_name)
                os.mkdir(cur_dir_path)

                with open(os.path.join(cur_dir_path, 'map.txt'), 'a') as f:
                    f.write('torch SEED: %d / cuda: %d\n' % (torch.initial_seed(), torch.cuda.initial_seed()))
                    f.write('==================================================\n')
                    f.write('...test map: map(i->t): %3.3f, map(t->i): %3.3f\n' % (mapi2t, mapt2i))
                    f.write('...test map: map(t->t): %3.3f, map(i->i): %3.3f\n' % (mapt2t, mapi2i))
                    f.write('==================================================\n')

                scipy.io.savemat(os.path.join(cur_dir_path, 'B_all.mat'), {
                    'BxTest': qBX,
                    'BxTrain': rBX,
                    'ByTest': qBY,
                    'ByTrain': rBY,
                    'LQuery': self.orig_data['L']['query'],
                    'LxDB': self.orig_data['L']['retrieval'],
                    'LyDB': self.orig_data['L']['retrieval'],
                })
                # save checkpoint
                state = {
                    'lnet': self.lnet.state_dict(),
                    'inet': self.model.inet.state_dict(),
                    'tnet': self.model.tnet.state_dict(),
                    'epoch': epoch
                }
                torch.save(state, os.path.join(cur_dir_path, self.checkpoint_path))

    def train_lab_net(self, var):
        print('update label_net')
        loss_total = 0.0
        for batch in tqdm(self.loaders['tloader_simple']):
            ind, image, text, label = batch
            clip_i = self.model.inet.get_clip_feature(image.cuda())
            clip_t = self.model.tnet.get_clip_feature(text.cuda())
            ind_batch, clip_pos, clip_neg = self.lnet(label.cuda())
            if ind_batch is None:
                print('skipped')
                continue
            clip_i = clip_i / clip_i.norm(dim=-1, keepdim=True)
            clip_t = clip_t / clip_t.norm(dim=-1, keepdim=True)
            clip_i_batch = clip_i[ind_batch]
            clip_t_batch = clip_t[ind_batch]
            clip_pos = clip_pos / clip_pos.norm(dim=-1, keepdim=True)
            clip_neg = clip_neg / clip_neg.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = clip_model.logit_scale.exp()
            logits_image_pos = logit_scale * (clip_i_batch * clip_pos).sum(1)
            logits_image_neg = logit_scale * (clip_i_batch * clip_neg).sum(1)
            logits_text_pos = logit_scale * (clip_t_batch * clip_pos).sum(1)
            logits_text_neg = logit_scale * (clip_t_batch * clip_neg).sum(1)

            loss_i = torch.clamp(logits_image_neg - logits_image_pos + self.margin, min=0).sum()
            loss_t = torch.clamp(logits_text_neg - logits_text_pos + self.margin, min=0).sum()
            loss_l = loss_i + loss_t
            loss_total += float(loss_l.detach().cpu().numpy())

            self.lnet_opt.zero_grad()
            loss_l.backward()
            self.lnet_opt.step()
        return loss_total

    def clip_pseudo_labeling(self):
        L_tr = self.orig_data['L']['train']
        n_sample, n_cls = L_tr.shape
        clip_i_all = torch.zeros(n_sample, 512, dtype=torch.float32).cuda()
        clip_t_all = torch.zeros(n_sample, 512, dtype=torch.float32).cuda()
        logit_scale = clip_model.logit_scale.exp()
        with torch.no_grad():
            for i_batch, batch in tqdm(enumerate(self.loaders['tloader_simple'])):
                ind, image, text, label = batch
                clip_i = self.model.inet.get_clip_feature(image.cuda())
                clip_t = self.model.tnet.get_clip_feature(text.cuda())
                #clip_i = clip_i / clip_i.norm(dim=-1, keepdim=True)
                #clip_t = clip_t / clip_t.norm(dim=-1, keepdim=True)
                clip_i_all[ind] = clip_i
                clip_t_all[ind] = clip_t
            #clip_l = self.model.tnet.get_clip_feature(tokenize(self.cfg.cls_vec, truncate=True).squeeze().cuda())
            clip_l = self.lnet.encode([[c] for c in range(n_cls)])
            logits_i = logit_scale * clip_i_all @ clip_l.t()
            # logits_t = logit_scale * clip_t_all @ clip_l.t()
            # logits = 0.5 * (logits_i + logits_t).squeeze()
            logits = logits_i.squeeze()
            for c in range(n_cls):
                logits_pos = logits[L_tr[:, c] == 1.0, c]
                logits_neg = logits[L_tr[:, c] == 0.0, c]
                logits_unk = logits[L_tr[:, c] == 0.0001, c]
                print(f'==== cls {c} ==========================================================')
                print(f'pos: mean = {logits_pos.mean().item()}, std = {logits_pos.std().item()}')
                print(f'neg: mean = {logits_neg.mean().item()}, std = {logits_neg.std().item()}')
                #clip_l = clip_l / clip_l.norm(dim=-1, keepdim=True)

    def clip_recovery_from_present(self, sel_modal='both', sel_thres=0.5):
        L_ic = self.orig_data['L']['ic_train']
        L_rec = np.zeros_like(L_ic)
        with torch.no_grad():
            for i_batch, batch in tqdm(enumerate(self.loaders['recloader'])):
                ind, image, text, label = batch
                cls_unknown = torch.where(label[0] == 0.0001)[0].numpy().tolist()
                if len(cls_unknown) == 0:
                    continue
                #print('ind: ', ind)
                n_cls = 1
                sel_cls_list_orig = torch.where(label[0] == 1.0)[0].numpy().tolist()
                if len(sel_cls_list_orig) > 13:
                    # L_rec[ind, sel_cls_list] = 1
                    continue
                sel_logit_list = []
                clip_i = self.model.inet.get_clip_feature(image.cuda())
                clip_t = self.model.tnet.get_clip_feature(text.cuda())
                clip_i = clip_i / clip_i.norm(dim=-1, keepdim=True)
                clip_t = clip_t / clip_t.norm(dim=-1, keepdim=True)

                clip_l = self.lnet.encode([sel_cls_list_orig])
                clip_l = clip_l / clip_l.norm(dim=-1, keepdim=True)
                logit_scale = clip_model.logit_scale.exp()
                if sel_modal == 'image':
                    logits = logit_scale * clip_i @ clip_l.t()
                elif sel_modal == 'text':
                    logits = logit_scale * clip_t @ clip_l.t()
                elif sel_modal == 'both':
                    logits_i = logit_scale * clip_i @ clip_l.t()
                    logits_t = logit_scale * clip_t @ clip_l.t()
                    logits = 0.5 * (logits_i + logits_t)
                else:
                    raise ValueError()
                # probs = logits.softmax(dim=-1).cpu().numpy()
                sel_cls_list = sel_cls_list_orig.copy()
                sel_logit_list.append(logits[0, 0].item())
                #while n_cls <= 24:
                while len(sel_cls_list) < 13:
                    #bfs_nodes = [sel_cls_list + [i_cls] for i_cls in range(L_rec.shape[1])]
                    bfs_nodes = [sel_cls_list + [i_cls] for i_cls in cls_unknown]
                    #print(bfs_nodes)
                    try:
                        clip_l = self.lnet.encode(bfs_nodes)
                    except:
                        print('error')
                    clip_l = clip_l / clip_l.norm(dim=-1, keepdim=True)
                    logit_scale = clip_model.logit_scale.exp()
                    if sel_modal == 'image':
                        logits = logit_scale * clip_i @ clip_l.t()
                    elif sel_modal == 'text':
                        logits = logit_scale * clip_t @ clip_l.t()
                    elif sel_modal == 'both':
                        logits_i = logit_scale * clip_i @ clip_l.t()
                        logits_t = logit_scale * clip_t @ clip_l.t()
                        logits = 0.5 * (logits_i + logits_t)
                    else:
                        raise ValueError()
                    # probs = logits.softmax(dim=-1).cpu().numpy()
                    values, cls_sort = logits.sort(dim=1, descending=True)
                    values = values[0]
                    cls_sort = cls_sort[0]
                    # print(values, cls_sort)
                    # print('cls_sort: ', cls_sort)
                    for i in range(cls_sort.shape[0]):
                        if cls_unknown[cls_sort[i].item()] not in sel_cls_list:
                            sel_cls_list.append(cls_unknown[cls_sort[i].item()])
                            sel_logit_list.append(values[i].item())
                            break
                    else:
                        break
                    sel_cls = sel_cls_list[-1]
                    #print("logit: ", sel_logit_list)
                    #print("sel_cls: ", sel_cls_list)
                    #print(desc_vec[sel_cls])
                    if len(sel_logit_list) > 1 and sel_logit_list[-1] < sel_logit_list[-2] + sel_thres:
                        sel_cls_list.pop(-1)
                        break
                    n_cls += 1
                L_rec[ind, sel_cls_list] = 1
                L_rec[ind, sel_cls_list_orig] = 0
        L_rec[L_ic == 0] = 0
        L_gt = self.orig_data['L']['gt_train']
        n_intersection = ((L_rec == 1) & (L_gt == 1)).sum()
        n_positive = (L_rec == 1).sum()
        #union = ((L_rec == 1) | (L_gt == 1)).sum()
        #iou = intersection / union
        #print(f'i: {intersection} / u: {union}, iou: {iou}')
        with open('./Ls_stat_log.txt', 'w') as f_log:
            f_log.write(f'tp: {n_intersection} / p: {n_positive}, prec: {n_intersection / n_positive}\n')
            print(f'tp: {n_intersection} / p: {n_positive}, prec: {n_intersection / n_positive}')
            cls_repr = self.cfg.cls_vec
            for i in range(L_rec.shape[1]):
                L_rec_i = L_rec[:, i]
                L_gt_i = L_gt[:, i]
                n_intersection = ((L_rec_i == 1) & (L_gt_i == 1)).sum()
                n_positive = (L_rec_i == 1).sum()
                #union = ((L_rec_i == 1) | (L_gt_i == 1)).sum()
                #iou = intersection / union
                #print(f'{i}-th class === i: {intersection} / u: {union}, iou: {iou}')
                f_log.write(f'class %2d === {cls_repr[i]}\n' % i)
                f_log.write(f'          === tp: {n_intersection} / p: {n_positive}, prec: {n_intersection / n_positive}\n')
                print(f'class %2d === {cls_repr[i]}' % i)
                print(f'          === tp: {n_intersection} / p: {n_positive}, prec: {n_intersection / n_positive}')
        L_ic_new = L_ic.copy()
        L_ic_new[L_rec == 1.0] = 1.0
        return L_ic_new

    def smooth_prob(self, logits, logit_orig, margin):
        return torch.sigmoid((logits - logit_orig) / margin * 4)

    def hard_prob(self, logits, logit_orig, margin):
        logits = (logits - logit_orig) / (margin * 0.5)
        logits = (torch.clamp(logits, -1, 1) + 1) / 2
        return logits

    def clip_recovery_from_present_one_step(self, sel_modal='both', margin=1.0):
        L_ic = self.orig_data['L']['ic_train']
        L_rec = np.ones_like(L_ic) * (-1)
        with torch.no_grad():
            for i_batch, batch in tqdm(enumerate(self.loaders['recloader'])):
                ind, image, text, label = batch
                cls_unknown = torch.where(label[0] == 0.0001)[0].numpy().tolist()
                if len(cls_unknown) == 0:
                    continue
                #print('ind: ', ind)
                n_cls = 1
                sel_cls_list_orig = torch.where(label[0] == 1.0)[0].numpy().tolist()
                if len(sel_cls_list_orig) > 13:
                    # L_rec[ind, sel_cls_list] = 1
                    continue
                sel_logit_list = []
                clip_i = self.model.inet.get_clip_feature(image.cuda())
                clip_t = self.model.tnet.get_clip_feature(text.cuda())
                clip_i = clip_i / clip_i.norm(dim=-1, keepdim=True)
                clip_t = clip_t / clip_t.norm(dim=-1, keepdim=True)

                clip_l = self.lnet.encode([sel_cls_list_orig])
                clip_l = clip_l / clip_l.norm(dim=-1, keepdim=True)
                logit_scale = clip_model.logit_scale.exp()
                if sel_modal == 'image':
                    logits = logit_scale * clip_i @ clip_l.t()
                elif sel_modal == 'text':
                    logits = logit_scale * clip_t @ clip_l.t()
                elif sel_modal == 'both':
                    logits_i = logit_scale * clip_i @ clip_l.t()
                    logits_t = logit_scale * clip_t @ clip_l.t()
                    logits = 0.5 * (logits_i + logits_t)
                else:
                    raise ValueError()
                # probs = logits.softmax(dim=-1).cpu().numpy()
                sel_cls_list = sel_cls_list_orig.copy()
                orig_logit = logits.item()

                #bfs_nodes = [sel_cls_list + [i_cls] for i_cls in range(L_rec.shape[1])]
                bfs_nodes = [[i_cls] + sel_cls_list for i_cls in cls_unknown]
                #print(bfs_nodes)
                try:
                    clip_l = self.lnet.encode(bfs_nodes)
                except:
                    print('error')
                clip_l = clip_l / clip_l.norm(dim=-1, keepdim=True)
                logit_scale = clip_model.logit_scale.exp()
                if sel_modal == 'image':
                    logits = logit_scale * clip_i @ clip_l.t()
                elif sel_modal == 'text':
                    logits = logit_scale * clip_t @ clip_l.t()
                elif sel_modal == 'both':
                    logits_i = logit_scale * clip_i @ clip_l.t()
                    logits_t = logit_scale * clip_t @ clip_l.t()
                    logits = 0.5 * (logits_i + logits_t)
                else:
                    raise ValueError()
                '''# probs = logits.softmax(dim=-1).cpu().numpy()
                values, cls_sort = logits.sort(dim=1, descending=True)
                values = values[0]
                cls_sort = cls_sort[0]
                # print(values, cls_sort)
                # print('cls_sort: ', cls_sort)
                for i in range(cls_sort.shape[0]):
                    if cls_unknown[cls_sort[i].item()] not in sel_cls_list:
                        sel_cls_list.append(cls_unknown[cls_sort[i].item()])
                        sel_logit_list.append(values[i].item())
                        break
                else:
                    break
                sel_cls = sel_cls_list[-1]
                #print("logit: ", sel_logit_list)
                #print("sel_cls: ", sel_cls_list)
                #print(desc_vec[sel_cls])
                if len(sel_logit_list) > 1 and sel_logit_list[-1] < sel_logit_list[-2] + sel_thres:
                    sel_cls_list.pop(-1)
                    break
                n_cls += 1'''
                L_rec[ind, cls_unknown] = self.hard_prob(logits, orig_logit, margin).cpu().numpy()
                #L_rec[ind, sel_cls_list_orig] = 0
        L_gt = self.orig_data['L']['gt_train']
        #n_intersection = ((L_rec == 1) & (L_gt == 1)).sum()
        #n_positive = (L_rec == 1).sum()
        rec_logit = []
        gt_logit = []
        tp_all = 0
        p_all = 0
        tn_all = 0
        n_all = 0
        with open('./Ls_stat_log.txt', 'w') as f_log:
            #f_log.write(f'tp: {n_intersection} / p: {n_positive}, prec: {n_intersection / n_positive}\n')
            #print(f'tp: {n_intersection} / p: {n_positive}, prec: {n_intersection / n_positive}')
            cls_repr = self.cfg.cls_vec
            for i in range(L_rec.shape[1]):
                f_log.write(f'class %2d === {cls_repr[i]}\n' % i)
                print(f'class %2d === {cls_repr[i]}' % i)
                rec_idx = np.where(L_rec[:, i] >= 0)[0]
                L_rec_i = L_rec[rec_idx, i]
                L_gt_i = L_gt[rec_idx, i]
                rec_logit.append(L_rec_i)
                gt_logit.append(L_gt_i)

                tp_i = ((L_rec_i == 1) & (L_gt_i == 1)).sum()
                p_i = ((L_rec_i == 1)).sum()
                tp_all += tp_i
                p_all += p_i
                f_log.write(f'          === tp: {tp_i} / p: {p_i}, prec: {tp_i / p_i}\n')
                print(f'          === tp: {tp_i} / p: {p_i}, prec: {tp_i / p_i}')

                tn_i = ((L_rec_i == 0) & (L_gt_i == 0)).sum()
                n_i = ((L_rec_i == 0)).sum()
                tn_all += tn_i
                n_all += n_i
                f_log.write(f'          === tn: {tn_i} / n: {n_i}, prec: {tn_i / n_i}\n')
                print(f'          === tn: {tn_i} / n: {n_i}, prec: {tn_i / n_i}')

                me_i_all = np.abs(L_rec_i - L_gt_i)
                pos_idx = np.where(L_gt_i == 1)
                neg_idx = np.where(L_gt_i == 0)
                me_i_pos = me_i_all[pos_idx].mean()
                me_i_neg = me_i_all[neg_idx].mean()
                me_i = me_i_all.mean()
                #union = ((L_rec_i == 1) | (L_gt_i == 1)).sum()
                #iou = intersection / union
                #print(f'{i}-th class === i: {intersection} / u: {union}, iou: {iou}')
                f_log.write(f'          === me: {me_i}, me_pos: {me_i_pos}, me_neg: {me_i_neg}\n')
                print(f'          === me: {me_i}, me_pos: {me_i_pos}, me_neg: {me_i_neg}\n')
            rec_logit = np.concatenate(rec_logit, axis=0)
            gt_logit = np.concatenate(gt_logit, axis=0)
            me_all = np.abs(rec_logit - gt_logit)
            pos_idx = np.where(gt_logit == 1)
            neg_idx = np.where(gt_logit == 0)
            me_pos = me_all[pos_idx].mean()
            me_neg = me_all[neg_idx].mean()
            f_log.write(f'overall:\n')
            print(f'overall:')
            f_log.write(f'          === tp: {tp_all} / p: {p_all}, prec: {tp_all / p_all}\n')
            print(f'          === tp: {tp_all} / p: {p_all}, prec: {tp_all / p_all}')
            f_log.write(f'          === tn: {tn_all} / n: {n_all}, prec: {tn_all / n_all}\n')
            print(f'          === tn: {tn_all} / n: {n_all}, prec: {tn_all / n_all}')
            f_log.write(f'          === me: {me_all.mean()}, me_pos: {me_pos}, me_neg: {me_neg}\n')
            print(f'          === me: {me_all.mean()}, me_pos: {me_pos}, me_neg: {me_neg}\n')
        L_ic_new = L_ic.copy()
        L_ic_new[L_rec >= 0] = L_rec[L_rec >= 0]
        return L_ic_new

    def our_loss(self, H, h, S):
        theta = 1.0 / 2 * torch.from_numpy(H).cuda().mm(h.transpose(1, 0))
        # prob = torch.sigmoid(theta)
        loss_pos = (S == 1) * (F.softplus(theta) - theta)
        loss_neg = (S == 0) * (F.softplus(theta))
        loss_pos = loss_pos.sum()
        loss_neg = loss_neg.sum()
        loss = loss_pos + loss_neg
        return loss, loss_pos, loss_neg

    def our_loss_continuous_sim_old(self, H, h, S):
        theta = 1.0 / 2 * torch.from_numpy(H).cuda().mm(h.transpose(1, 0))
        # prob = torch.sigmoid(theta)
        loss_pos = (S > 0) * (F.softplus(theta) - S * theta)
        loss_neg = (S == 0) * (F.softplus(theta))
        loss_pos = loss_pos.sum()
        loss_neg = loss_neg.sum()
        loss = loss_pos + loss_neg
        return loss, loss_pos, loss_neg

    def our_loss_continuous_sim(self, H, h, S):
        theta = 1.0 / 2 * torch.from_numpy(H).cuda().mm(h.transpose(1, 0))
        # prob = torch.sigmoid(theta)
        loss = (S >= 0) * (F.softplus(theta) - S * theta)
        loss = loss.sum()
        return loss

    def loss_focal_margin_unknown(self, H, h, S, epoch, i_iter,
            tau: float = 0.8,
            margin: float = 1.0,  # 应调参数
            gamma: float = 2.0,):
        # theta = 1.0 / 2 * torch.from_numpy(H).cuda().mm(h.transpose(1, 0))
        theta = 1.0 / 2 * torch.from_numpy(H).cuda().mm(h.transpose(1, 0))
        # prob = torch.sigmoid(theta)

        prob = torch.sigmoid(theta).detach()
        if i_iter % 500 == 0:
            pos_data = torch.masked_select(prob, S.bool())
            neg_data = torch.masked_select(prob, ~S.bool())
            print(
                f'[iter {i_iter}] P_mean: {pos_data.mean().data} | N_mean: {neg_data.mean().data} | P_std: {pos_data.std().data} | N_std: {neg_data.std().data}')
        theta = torch.where(S == 1, theta - margin, theta)
        pred = torch.sigmoid(theta)
        # Focal margin for postive loss
        pt = (1 - pred) * S + pred * (1 - S)
        focal_weight = pt ** self.gamma

        loss_pos = -(S == 1) * F.logsigmoid(theta)
        loss_neg = -(S == 0) * F.logsigmoid(-theta)
        loss_pos = (focal_weight * loss_pos).sum()
        loss_neg = (focal_weight * loss_neg).sum()
        loss = loss_pos + loss_neg
        return loss, loss_pos, loss_neg

    def loss_focal_margin(self, H, h, S, epoch, i_iter,
            tau: float = 0.8,
            margin: float = 1.0,
            gamma: float = 2.0,):
        # theta = 1.0 / 2 * torch.from_numpy(H).cuda().mm(h.transpose(1, 0))
        theta = 1.0 / 2 * torch.from_numpy(H).cuda().mm(h.transpose(1, 0))
        # prob = torch.sigmoid(theta)

        prob = torch.sigmoid(theta).detach()
        indicator = prob > tau
        if i_iter % 500 == 0:
            pos_data = torch.masked_select(prob, S.bool())
            neg_data = torch.masked_select(prob, ~S.bool())
            n_changed = ((1 - S) * indicator).sum(dim=0).mean().data
            print(
                f'[iter {i_iter}] silenced: {n_changed} | P_mean: {pos_data.mean().data} | N_mean: {neg_data.mean().data} | P_std: {pos_data.std().data} | N_std: {neg_data.std().data}')
        theta = torch.where(S == 1, theta - margin, theta)
        pred = torch.sigmoid(theta)
        # Focal margin for postive loss
        pt = (1 - pred) * S + pred * (1 - S)
        focal_weight = pt ** self.gamma

        loss_pos = -S * F.logsigmoid(theta)
        loss_neg = -(1 - S) * F.logsigmoid(-theta)
        loss_pos = (focal_weight * loss_pos).sum()
        loss_neg = (focal_weight * loss_neg).sum()
        loss = loss_pos + loss_neg
        return loss, loss_pos, loss_neg

    def loss_focal_margin_hn_ignore(self, H, h, S, epoch, i_iter,
            tau: float = 0.8,
            change_epoch: int = 15,
            margin: float = 1.0,
            gamma: float = 2.0,):
        # theta = 1.0 / 2 * torch.from_numpy(H).cuda().mm(h.transpose(1, 0))
        theta = 1.0 / 2 * torch.from_numpy(H).cuda().mm(h.transpose(1, 0))
        # prob = torch.sigmoid(theta)

        prob = torch.sigmoid(theta).detach()
        indicator = prob > tau
        if i_iter % 500 == 0:
            pos_data = torch.masked_select(prob, S.bool())
            neg_data = torch.masked_select(prob, ~S.bool())
            n_changed = ((1 - S) * indicator).sum(dim=0).mean().data
            print(
                f'[iter {i_iter}] silenced: {n_changed} | P_mean: {pos_data.mean().data} | N_mean: {neg_data.mean().data} | P_std: {pos_data.std().data} | N_std: {neg_data.std().data}')
        if epoch >= change_epoch:
            M = indicator * (1 - S)
        else:
            M = (1 - S)
        theta = torch.where(S == 1, theta - margin, theta)
        pred = torch.sigmoid(theta)
        # Focal margin for postive loss
        pt = (1 - pred) * S + pred * (1 - S)
        focal_weight = pt ** self.gamma

        loss_pos = -S * F.logsigmoid(theta)
        loss_neg = -M * F.logsigmoid(-theta)
        loss_pos = (focal_weight * loss_pos).sum()
        loss_neg = (focal_weight * loss_neg).sum()
        loss = loss_pos + loss_neg
        return loss, loss_pos, loss_neg

    def mixed_label_old(self, label, label2):
        bs = label.shape[0]
        label_mixed = 0.2 * label + 0.8 * label2
        label_mixed = torch.where((label_mixed < 0.0001) & (label_mixed > 0.0), 0.0001, label_mixed)
        return label_mixed

    def mixed_label_old2(self, label, label2):
        bs = label.shape[0]
        label_mixed = label + label2
        label_mixed = torch.where((label_mixed < 0.01) & (label_mixed > 0.0), 0.0001, label_mixed)
        label_mixed = torch.where((label_mixed > 1), 1.0, label_mixed)
        return label_mixed

    def mixed_label(self, label, label2, coef):
        #label[label == 0.0001] = 0.1
        #label2[label2 == 0.0001] = 0.1
        label_mixed = coef * label + (1 - coef) * label2
        # label_mixed = torch.where((label_mixed < 0.0001) & (label_mixed > 0.0), 0.0001, label_mixed)
        return label_mixed

    def train_both_net(self, var):
        print('update text_net')
        V = var['v']
        T = var['t']
        B = var['B']
        loss_total = 0.0
        L_tr_cuda = torch.from_numpy(self.orig_data['L']['train']).cuda()
        # loader2 = iter(self.loaders['tloader2'])
        for i, batch in enumerate(self.loaders['tloader']):
            ind, image, text, label, image2, text2, label2 = batch
            # try:
            #    batch2 = next(loader2)
            # except StopIteration:
            #    break
            # ind2, image2, text2, label2 = batch2
            hsh_i, hsh_t, hsh_i_mix, hsh_t_mix = self.model(image.cuda(), text.cuda(), image2.cuda(), text2.cuda())
            # hsh_i, hsh_t = self.model(image.cuda(), text.cuda(), image2.cuda(), text2.cuda())
            V[ind, :] = hsh_i.detach().cpu().numpy()
            T[ind, :] = hsh_t.detach().cpu().numpy()

            mean_coef = (self.model.inet.get_coef() + self.model.tnet.get_coef()) / 2
            label_mixed = self.mixed_label(label.cuda(), label2.cuda(), mean_coef)
            #print(label_mixed)

            S = calc_neighbor_continuous_label_pytorch(L_tr_cuda, label.cuda())
            S_aug = calc_neighbor_continuous_label_pytorch(L_tr_cuda, label_mixed.cuda())

            b = torch.from_numpy(B[ind, :]).cuda()

            #loss_it, loss_it_pos, loss_it_neg = self.loss_focal_margin(V, hsh_t, S, var['epoch'], i)
            #loss_ti, loss_ti_pos, loss_ti_neg = self.loss_focal_margin(T, hsh_i, S, var['epoch'], i)
            loss_it = self.our_loss_continuous_sim(V, hsh_t, S)
            loss_ti = self.our_loss_continuous_sim(T, hsh_i, S)

            loss_it_aug = self.our_loss_continuous_sim(V, hsh_t_mix, S_aug)
            loss_ti_aug = self.our_loss_continuous_sim(T, hsh_i_mix, S_aug)

            if i % 250 == 0:
                #print(f'>>>>>>>>epoch.{var["epoch"]} - loss_sim: {loss_it.data + loss_ti.data}, pos: {loss_it_pos.data + loss_ti_pos.data}, neg: {loss_it_neg.data + loss_ti_neg.data}.')
                print(f'>>>>>>>>epoch.{var["epoch"]} - loss_sim: {loss_it.data + loss_ti.data}.')
                #print(f'>>>>>>>>epoch.{var["epoch"]} - loss_aug: {loss_it_aug.data + loss_ti_aug.data}, pos: {loss_it_pos_aug.data + loss_ti_pos_aug.data}, neg: {loss_it_neg_aug.data + loss_ti_neg_aug.data}.')
                print(f'>>>>>>>>epoch.{var["epoch"]} - loss_aug: {loss_it_aug.data + loss_ti_aug.data}.')
                print(f'>>>>>>>>epoch.{var["epoch"]} - coef_im: {self.model.inet.get_coef().item()}, coef_im: {self.model.tnet.get_coef().item()}')
            loss_quant = F.mse_loss(b, hsh_i, reduction='sum') + F.mse_loss(b, hsh_t, reduction='sum')\
                       + F.mse_loss(torch.sign(hsh_i_mix).detach(), hsh_i_mix, reduction='sum') + F.mse_loss(torch.sign(hsh_t_mix).detach(), hsh_t_mix, reduction='sum')
            #loss_quant = self.loss_quant_triplet(hsh_i, hsh_t, label.cuda())
            loss = loss_it + loss_ti + loss_it_aug + loss_ti_aug + self.alpha * loss_quant
            #loss = loss_it + loss_ti + self.alpha * loss_quant
            loss_total += float(loss.detach().cpu().numpy())

            self.inet_opt.zero_grad()
            self.tnet_opt.zero_grad()
            loss.backward()
            self.inet_opt.step()
            self.tnet_opt.step()
        return loss_total

    def calc_isfrom_acc(self, train_isfrom_, Train_ISFROM):
        erro = Train_ISFROM.shape[0] - np.sum(
            np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype(int))
        acc = np.divide(np.sum(np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype('float32')),
                        Train_ISFROM.shape[0])
        return erro, acc

    def generate_code(self, loader):
        num_data = len(loader.dataset)
        #ind_shift = loader.dataset.ind_shift
        BX = np.zeros([num_data, self.bit], dtype=np.float32)
        BY = np.zeros([num_data, self.bit], dtype=np.float32)
        for batch in tqdm(loader):
            ind, image, text, label = batch
            #ind = ind - ind_shift
            hsh_i = self.model.get_image_hash(image.cuda())
            BX[ind, :] = hsh_i.cpu().numpy()
            hsh_t = self.model.get_text_hash(text.cuda())
            BY[ind, :] = hsh_t.cpu().numpy()
        BX = np.sign(BX)
        BY = np.sign(BY)
        return BX, BY
