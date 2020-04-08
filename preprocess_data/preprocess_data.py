import os
data_path=os.path.join('../data','COVID-Dialogue-Dataset-English.txt')
data_dest_source=os.path.join('patient2doctor','train.source')
data_dest_target=os.path.join('patient2doctor','train.target')

Dialogue_on=False
Patient_on=False
Doctor_on=False
Patient_input=''
Doctor_output=''
Patient_idx=0
Doctor_idx=0
with open(data_path) as f, open(data_dest_source,'w') as source, open(data_dest_target,'w') as target:
    for line in f.readlines():
        line=line.strip()
        if line=='Dialogue':
            Dialogue_on=True
            Patient_on=False
            Doctor_on=False
            if Patient_input!='':
                source.write(Patient_input+'\n')
                Patient_idx+=1
                Patient_input=''
                assert Doctor_output==''
                target.write(' \n')
                Doctor_idx+=1
            elif Doctor_output!='':
                target.write(Doctor_output+'\n')
                Doctor_idx+=1
                if Doctor_idx!=Patient_idx:
                    raise AssertionError(Doctor_output)
                Doctor_output=''
                assert Patient_input==''
        elif line=='Patient:':
            Patient_on=True
            Doctor_on=False
            Patient_input=''
            if Doctor_output!='':
                target.write(Doctor_output+'\n')
                Doctor_idx+=1
                if Doctor_idx!=Patient_idx:
                    raise AssertionError(Doctor_output)
                Doctor_output=''
        elif line=='Doctor:':
            Doctor_on=True
            Patient_on=False
            Doctor_output=''
            if Patient_input!='':
                source.write(Patient_input+'\n')
                Patient_input=''
                Patient_idx+=1
        elif 'id=' in line and len(line)<10:
            Dialogue_on=False
            Patient_on=False
            Doctor_on=False
        else:
            if Dialogue_on and Patient_on:
                Patient_input=Patient_input+line+' '
            elif Dialogue_on and Doctor_on:
                line=line.replace('In brief:','')
                line=line.split()
                for i in range(len(line)):
                    if 'https://' in line[i]:
                        line[i]='https://www.healthtap.com/blog/covid-19-care-guidelines'
                line=' '.join(line)
                Doctor_output=Doctor_output+line+' '
    #for loop ended
    if Patient_input!='':
        source.write(Patient_input+'\n')
        Patient_input=''
        assert Doctor_output==''
        target.write(' \n')
    elif Doctor_output!='':
        target.write(Doctor_output+'\n')
        Doctor_output=''
        assert Patient_input==''


f.close()
source.close()
target.close()