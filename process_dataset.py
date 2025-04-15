import os
import glob
import numpy as np
import pandas as pd

def preprocess_mimic_data():
    patient_df = pd.read_csv('/secure/shared_data/SOAP/MIMIC/PATIENTS.csv').drop(
        columns=['ROW_ID', 'DOD_HOSP', 'DOD_SSN']
    )
    patient_df['DOB'] = pd.to_datetime(patient_df['DOB'])
    patient_df['DOD'] = pd.to_datetime(patient_df['DOD'])

    admission_df = pd.read_csv('/secure/shared_data/SOAP/MIMIC/ADMISSIONS.csv')
    admission_df = admission_df[admission_df['ADMISSION_TYPE'] != 'NEWBORN']
  
    admission_df = admission_df.drop(
        columns=['ROW_ID', 'ADMISSION_LOCATION', 'DISCHARGE_LOCATION',
                 'INSURANCE', 'EDREGTIME', 'EDOUTTIME',
                 'HOSPITAL_EXPIRE_FLAG', 'HAS_CHARTEVENTS_DATA']
    )
 
    admission_df['ADMITTIME'] = pd.to_datetime(admission_df['ADMITTIME'])
    admission_df['DISCHTIME'] = pd.to_datetime(admission_df['DISCHTIME'])

    admission_df = admission_df.sort_values(by=['SUBJECT_ID', 'ADMITTIME']).reset_index(drop=True)
    admission_df['NEXT_ADMITTIME'] = admission_df.groupby('SUBJECT_ID')['ADMITTIME'].shift(-1)

    admission_df = admission_df.merge(
        patient_df[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD', 'EXPIRE_FLAG']],
        on='SUBJECT_ID',
        how='left'
    )

    admission_df['AGE_AT_ADMISSION'] = admission_df.apply(
        lambda row: row['ADMITTIME'].year - row['DOB'].year
                    - ((row['ADMITTIME'].month, row['ADMITTIME'].day)
                       < (row['DOB'].month, row['DOB'].day)),
        axis=1
    )

    admission_df['SUBJECT_ID'] = admission_df['SUBJECT_ID'].astype(str)
    admission_df['HADM_ID'] = admission_df['HADM_ID'].astype(str)

    note_df = pd.read_csv(
        '/secure/shared_data/SOAP/MIMIC/NOTEEVENTS.csv',
        engine='python',
        on_bad_lines='skip',
        sep=',',
        dtype=str
    )
    note_df = note_df[note_df['HADM_ID'].isin(admission_df['HADM_ID'])]

    expanded_admission_df = admission_df.merge(
        note_df[['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME', 'CATEGORY', 'DESCRIPTION', 'TEXT']],
        on=['SUBJECT_ID', 'HADM_ID'],
        how='left'
    )
    expanded_admission_df['CHARTDATE'] = pd.to_datetime(expanded_admission_df['CHARTDATE'], errors='coerce')
    expanded_admission_df['CHARTTIME'] = pd.to_datetime(expanded_admission_df['CHARTTIME'], errors='coerce')

    expanded_admission_df = expanded_admission_df.sort_values(
        by=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME']
    ).reset_index(drop=True)


    df = expanded_admission_df.copy()
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'], errors='coerce')
    df['DISCHTIME'] = pd.to_datetime(df['DISCHTIME'], errors='coerce')
    df['DOD'] = pd.to_datetime(df['DOD'], errors='coerce')

    # 90-Day Readmission
    df['90_DAY_READMISSION'] = 0
    mask_90_day = (
        (df['NEXT_ADMITTIME'] - df['DISCHTIME']).notna() &
        ((df['NEXT_ADMITTIME'] - df['DISCHTIME']).dt.days <= 90) &
        ((df['NEXT_ADMITTIME'] - df['DISCHTIME']).dt.days >= 0)
    )
    df.loc[mask_90_day, '90_DAY_READMISSION'] = 1

    # 1-Year Mortality
    df['1_YEAR_MORTALITY'] = 0
    mask_1yr = (
        df['DOD'].notna() &
        ((df['DOD'] - df['DISCHTIME']).dt.days <= 365) &
        ((df['DOD'] - df['DISCHTIME']).dt.days >= 0)
    )
    df.loc[mask_1yr, '1_YEAR_MORTALITY'] = 1

    df_final = df[df['DEATHTIME'].isna()]
    df_final.to_csv('mimic_df.csv', index=False)
    return df_final


def process_glaucoma_data():
    gender_map = {0: "Female", 1: "Male"}
    race_map = {0: "Asian", 1: "Black", 2: "White"}
    ethnicity_map = {0: "non-Hispanic", 1: "Hispanic", -1: "Unknown"}
    language_map = {0: "English", 1: "Spanish", 2: "Other", -1: "Unknown"}
    maritalstatus_map = {
        0: "Marriage or Partnered", 1: "Single", 2: "Divorced",
        3: "Widowed", 4: "Legally Separated", -1: "Unknown"
    }
    glaucoma_map = {0: "Non-Glaucoma", 1: "Glaucoma"}

    file_list = glob.glob('/secure/xiaoyang/HarvardFairVLMed/Dataset/Test/*.npz')
    print(f"Found {len(file_list)} glaucoma .npz files.")

    df_list = []

    for file_path in file_list:
        data = np.load(file_path)

        ms_val = data["maritalstatus"].item()
        eth_val = data["ethnicity"].item()
        lang_val = data["language"].item()
        gen_val  = data["gender"].item()
        race_val = data["race"].item()
        age_val  = data["age"].item()
        glu_val  = data["glaucoma"].item()

        ms_str  = maritalstatus_map[ms_val]
        eth_str = ethnicity_map[eth_val]
        lang_str= language_map[lang_val]
        gen_str = gender_map[gen_val]
        race_str= race_map[race_val]
        glu_str = glaucoma_map[glu_val]

        slo_fundus_array = data["slo_fundus"]

        row_df = pd.DataFrame({
            "slo_fundus": [slo_fundus_array],
            "maritalstatus": [ms_str],
            "ethnicity": [eth_str],
            "language": [lang_str],
            "gender": [gen_str],
            "race": [race_str],
            "age": [age_val],
            "glaucoma": [glu_str]
        })
        df_list.append(row_df)

    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv('glaucoma_df.csv', index=False) 
    return final_df


def main():
    preprocess_mimic_data()

    process_glaucoma_data()

    print("Done.")


if __name__ == "__main__":
    main()
