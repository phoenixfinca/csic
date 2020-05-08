import re
import numpy as np
normal_file_pre =r'C:\Users\kk\Desktop\dataset\CSIC2010\\processed\normal.txt'
anomalous_file_pre = r'C:\Users\kk\Desktop\dataset\CSIC2010\\processed\anomalous.txt'


vector_normal = r'C:\Users\kk\Desktop\dataset\CSIC2010\processed\vector_normal.txt'
vector_anomalous = r'C:\Users\kk\Desktop\dataset\CSIC2010\processed\vector_abnormal.txt'


#str = 'http://localhost:8080/tienda1/publico/anadir.jsp?id=3&nombre=vino+rioja&precio=100&cantidad=55&b1=aadir+al+carrito'

def vector(data_link):
    per_fea= []
    url_len= len(data_link)
    per_fea.append(url_len)
    s = data_link.split("?",1)
    if len(s)!= 1:#判断是否有参数
        par_len= len(s[1])#s[0]是路径，s[1]是整个参数
        par= s[1].split("&")#&分隔各个参数
        par_num= len(par)#par是参数数组
        #初始化
        par_max_l= 0#最大长度
        number_num= 0#数字个数
        str_num= 0#字符个数
        spe_num= 0#特殊字符个数
        par_val_sum = 0#参数总长度
        for pa in par:
            try:
                [par_name,par_val] = pa.split("=",1)#par_name是参数名称，par_val是参数内容，重要的是par_val
                par_val_sum =  par_val_sum + len(par_val)
                if par_max_l<len(par_val):
                    par_max_l= len(par_val)
                # pdb.set_trace()

                num_regex = re.compile(r'\d')
                zimu_regex = re.compile(r'[a-zA-Z]')

                number_num= number_num+len(num_regex.findall(par_val))
                str_num= str_num+len(zimu_regex.findall(par_val))
                spe_num= spe_num+len(par_val) -len(num_regex.findall(par_val)) -len(zimu_regex.findall(par_val))
            except ValueError as err:
                continue
        try:
            number_rt= number_num/par_val_sum
            str_rt= str_num/par_val_sum
            spe_rt= spe_num/par_val_sum
        except ZeroDivisionError as err:
            number_rt = 0
            str_rt = 0
            spe_rt = 0
        per_fea = np.append(per_fea, [par_len, par_num, par_max_l, number_num, number_rt, str_rt, spe_num, spe_rt])
    else:
        per_fea = np.append(per_fea, [0, 0, 0, 0, 0, 0, 0, 0])
    return per_fea


def process_data(file_in,file_out=None):
    with open(file_in, 'r', encoding='utf-8') as f:
        data_lines = f.readlines()
        result = []
        for d in data_lines:
            data=d.strip()
            data_vector = vector(data)
            if len(data_vector) > 0:
                result.append(data_vector)

    np.savetxt(file_out,result,fmt='%.2f')
    # with open(file_out, 'w', encoding='utf-8') as f_out:
    #     for line in result:
    #         f_out.writelines(line + '\n')


if __name__  == '__main__':
    process_data(normal_file_pre,vector_normal)
    process_data(anomalous_file_pre,vector_anomalous)


#data = vector(str)
