"""General 3D topology optimization code by Zhihao Zuo and Yimin Xie. Note that the CAE file shall contain a model 'Model-1' with a dependent part 'Part-1' and a static step 'Step-1'."""
import math,customKernel
from abaqus import getInput,getInputs
from odbAccess import openOdb
## Function of formatting Abaqus model for stiffness optimisation
def fmtMdb(Mdb):
    mdl = Mdb.models['Model-1']
    part = mdl.parts['Part-1']
    # Build sections and assign solid section
    mdl.Material('Material01').Elastic(((1.0, 0.3), ))
    mdl.HomogeneousSolidSection('sldSec','Material01')
    mdl.Material('Material02').Elastic(((0.001**3, 0.3), ))
    mdl.HomogeneousSolidSection('voidSec','Material02')
    part.SectionAssignment(part.Set('ss',part.elements),'sldSec')
    # Define output request
    mdl.FieldOutputRequest('SEDensity','Step-1',variables=('ELEDEN', ))
    mdl.HistoryOutputRequest('ExtWork','Step-1',variables=('ALLWK', ))
## Function of running FEA for raw sensitivities and objective function
def FEA(Iter,Mdb,Xe,Ae):
    # 提交job，并等待job完成，打开结果odb提取ESEDEN(元素的Von Mise)
    Mdb.Job('Design_Job'+str(Iter),'Model-1', numCpus=10, numDomains=10).submit()
    Mdb.jobs['Design_Job'+str(Iter)].waitForCompletion()
    opdb = openOdb('Design_Job'+str(Iter)+'.odb')
    seng = opdb.steps['Step-1'].frames[-1].fieldOutputs['ESEDEN'].values
    for en in seng: 
        # 遍历ESEDEN的每个值en，使用元素的应力en.data除以其密度Xe[en.elementLabel]来计算敏感度，并将结果存储在Ae数组中，索引为元素的标签en.elementLabel
        Ae[en.elementLabel]=en.data/Xe[en.elementLabel]
    # 获取历程输出中ALLWK（总功）作为柔度
    obj=opdb.steps['Step-1'].historyRegions['Assembly ASSEMBLY'].historyOutputs['ALLWK'].data[-1][1]
    opdb.close()
    return obj
## Function of preparing filter map (Fm={elm1:[[el1,el2,...],[wf1,wf2,...]],...})
def preFlt(Rmin,Elmts,Nds,Fm):
    c0 = {}# 存储每个元素的质心坐标，键是元素的标签 el.label，值是元素的质心坐标（列表类型，列表元素是质心的xyz坐标轴值）
    for el in Elmts:
        nds = el.connectivity
        c0[el.label]=[sum([Nds[nd].coordinates[i]/len(nds) for nd in nds]) for i in range(3)]
    # 计算权重因子：1）遍历Elmts，对每个element在过滤图Fm中初始化两个空列表，第一个列表Fm[el.label][0]存储于当前元素相邻的元素的label，第二个列表Fm[el.label][1]存储相应的权重因子
    for el in Elmts:
        Fm[el.label] = [[],[]]
        # 计算element间的欧几里得距离：通过取两个元素质心坐标差的平方和的平方根来实现的，质心坐标存储在之前初始化的 c0 字典中。
        for em in Elmts:
            dis=math.sqrt(sum([(c0[el.label][i]-c0[em.label][i])**2 for i in range(3)]))
            # 确定相邻元素并计算权重：过滤半径与实际距离的差值作为权重添加到 Fm[el.label][1] 列表中。这个权重表示元素 em 对 el 的影响程度，距离越近，权重越大。
            if dis<Rmin:
                Fm[el.label][0].append(em.label)
                Fm[el.label][1].append(Rmin - dis)
        # 归一化权重因子
        sm = sum(Fm[el.label][1])
        for i in range(len(Fm[el.label][0])): Fm[el.label][1][i] /= sm
## 过滤灵敏度：目的是平滑设计域中的灵敏度分布，避免优化中出现棋盘格
def fltAe(Ae,Fm):
    # Ae存储了每个element的原始灵敏度，Fm字典存储了用于过滤操作的权重信息
    raw = Ae.copy()
    for el in Fm.keys():
        Ae[el] = 0.0# 对于当前元素el，将其在Ae中的敏感度值初始化为零。因为将通过累加的方式重新计算过滤后的敏感度值。
        for i in range(len(Fm[el][0])): 
            # Fm[el][0] 是一个列表，包含了相邻元素的标签；Fm[el][1] 是一个列表，包含了对应的权重因子。
            # 相邻元素 Fm[el][0][i] 的原始敏感度值*获取与相邻元素对应的权重因子----将加权原始敏感度值累加到当前元素 el 的敏感度上
            Ae[el]+=raw[Fm[el][0][i]]*Fm[el][1][i]
## Function of optimality update for design variables and Abaqus model
def BESO(Vf,Xe,Ae,Part,Elmts):
    lo, hi = min(Ae.values()), max(Ae.values())# 计算敏感度的最小值和最大值
    tv = Vf*len(Elmts)# 计算目标体积 tv，它是目标体积分数 Vf 与模型中元素总数的乘积
    # 二分搜索以确定阈值  直到敏感度的上下限差异小于一个很小的容忍度（1.0e-5）
    while (hi-lo)/hi > 1.0e-5:
        th = (lo+hi)/2.0# 计算 lo 和 hi 的平均值作为当前的阈值 th

        # 遍历所有设计变量 Xe，如果元素的敏感度 Ae[key] 大于阈值 th，则将该元素的设计变量设置为1.0，否则设置为0.001
        for key in Xe.keys(): Xe[key] = 1.0 if Ae[key]>th else 0.001
        
        # 计算当前设计变量的总和与目标体积 tv 的差值。如果总和大于目标体积，则减小阈值（将 lo 设置为 th），否则增加阈值（将 hi 设置为 th）
        if sum(Xe.values())-tv>0: lo = th
        else: hi = th

    # 初始化两个列表 vlb 和 slb，分别用于存储空洞元素和实体元素的标签
    vlb, slb = [], []
    for el in Elmts:
        if Xe[el.label] == 1.0: slb.append(el.label)
        else: vlb.append(el.label)

    # 分配材料属性截面到实体和空洞元素
    Part.SectionAssignment(Part.SetFromElementLabels('ss',slb),'sldSec')
    Part.SectionAssignment(Part.SetFromElementLabels('vs',vlb),'voidSec')
## ====== MAIN PROGRAM ======
if __name__ == '__main__':
    # Set parameters and inputs
    pars = (('VolFrac:','0.5'), ('Rmin:', '1'), ('ER:', '0.02'))
    vf,rmin,ert = [float(k) if k!=None else 0 for k in getInputs(pars,dialogTitle='Parameters')]
    if vf<=0 or rmin<0 or ert<=0: sys.exit()

    # Design initialization
    mddb = openMdb(getInput('Input CAE file:',default='Test.cae'))# 生成cae模型数据库对象mddb
    fmtMdb(mddb)# 调用函数对mddb对象进行初始化处理：定义实体、空洞元素的材料属性（初始全部为实体）；定义场输出ELEDEN-单元应变能、历程输出ALLWK-等效于考虑无能量耗散的柔度目标函数
    part = mddb.models['Model-1'].parts['Part-1']# 将mddb对象中经常要用到的part、elements、nodes信息赋值给变量
    elmts, nds = part.elements, part.nodes
    oh, vh = [], []# 存储柔度目标函数、体积的历史迭代信息
    xe, ae, oae, fm = {}, {}, {}, {}# 存储拓扑设计变量、当前element的灵敏度、上一次迭代的element灵敏度、过滤方案的element映射

    for el in elmts: xe[el.label] = 1.0# 初始的拓扑设计变量全为1，即全为solid element
    if rmin>0: preFlt(rmin,elmts,nds,fm)# 存在有效过滤半径时，触发过滤方案preFlt函数：计算权重因子

    # change作为目标函数的收敛误差；iter迭代次数；obj是
    change, iter, obj = 1, -1, 0
    while change > 0.001:
        iter += 1
        # 调用有限元分析函数，传入（迭代次数，cae模型数据库对象，拓扑设计变量，当前迭代中的element灵敏度），返回
        # 返回结果存入记录柔度目标函数历史信息的列表oh
        oh.append(FEA(iter,mddb,xe,ae))

        # 过滤灵敏度，并对过滤后的灵敏度进行历史平均，记录本次迭代中结果两次处理得到的灵敏度
        if rmin>0: 
            fltAe(ae,fm)
        if iter > 0: 
            ae=dict([(k,(ae[k]+oae[k])/2.0) for k in ae.keys()])
        oae = ae.copy()# 记录本次迭代的灵敏度，以供下一次迭代中进行历史平均

        # BESO优化：
        vh.append(sum(xe.values())/len(xe))# 记录当前设计变量对应的体积分数
        nv = max(vf,vh[-1]*(1.0-ert))# 根据进化比计算下一次迭代的体积分数
        BESO(nv,xe,ae,part,elmts)# 根据本轮迭代目标体积分数，拓扑设计变量，单元灵敏度，part，elmts，进行beso单元增删

        # 每迭代11轮，计算优化收敛性的变化 ，当change很小，说明目标函数值变化已不大，可能已经接近收敛
        if iter>10: change=math.fabs((sum(oh[iter-4:iter+1])-sum(oh[iter-9:iter-4]))/sum(oh[iter-9:iter-4]))

    # Save results
    mddb.customData.History = {'vol':vh,'obj':oh}
    mddb.saveAs('Final_design.cae')