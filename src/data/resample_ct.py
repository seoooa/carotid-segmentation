import os
import argparse
import numpy as np
import nibabel as nib
from monai.transforms import Resize
import torch

def resample_images(input_base_dir, output_base_dir, target_size=(1024, 1024), start_case=1, end_case=42):
    """
    모든 케이스의 CT 이미지와 세그멘테이션 파일을 1024x1024 크기로 리샘플링합니다.
    
    Args:
        input_base_dir: 원본 파일이 있는 기본 디렉토리 경로
        output_base_dir: 리샘플링된 파일을 저장할 기본 디렉토리 경로
        target_size: 리샘플링할 대상 크기 (가로, 세로)
        start_case: 시작 케이스 번호
        end_case: 종료 케이스 번호
    """
    # 출력 기본 디렉토리가 없으면 생성
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 각 케이스 처리
    for case_num in range(start_case, end_case + 1):
        case_id = f"case_{case_num:02d}"  # 예: case_01, case_02, ...
        patient_num = f"{case_num:02d}"  # 환자 번호만 추출 (예: "01")
        
        # 입력 파일 경로
        input_dir = os.path.join(input_base_dir, case_id)
        
        # CT 이미지 파일
        ct_file = os.path.join(input_dir, f"CT.nii.gz")
        
        # 세그멘테이션 파일들
        carotid_file = os.path.join(input_dir, f"carotid.nii.gz")
        combined_seg_file = os.path.join(input_dir, f"combined_seg.nii.gz")
        
        # 출력 파일 경로
        output_dir = os.path.join(output_base_dir, case_id)
        output_ct_file = os.path.join(output_dir, "CT.nii.gz")
        output_carotid_file = os.path.join(output_dir, f"carotid.nii.gz")
        output_combined_seg_file = os.path.join(output_dir, f"combined_seg.nii.gz")
        
        # 입력 파일이 존재하는지 확인
        if not os.path.exists(ct_file):
            print(f"경고: CT 이미지 파일을 찾을 수 없습니다 - {ct_file}")
            continue
        
        # 출력 디렉토리가 없으면 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # CT 이미지 로드
        print(f"처리 중: {case_id}")
        ct_img = nib.load(ct_file)
        ct_data = ct_img.get_fdata()
        affine = ct_img.affine
        
        # 현재 이미지 크기 확인
        current_shape = ct_data.shape
        print(f"  현재 크기: {current_shape}")
        
        # 크기 조정이 필요한지 확인
        if current_shape[0] != target_size[0] or current_shape[1] != target_size[1]:
            print(f"  {target_size} 크기로 리샘플링 중...")
            
            # 1. CT 이미지 리샘플링 (bilinear 보간)
            print("  CT 이미지 리샘플링 중...")
            ct_tensor = torch.from_numpy(ct_data).unsqueeze(0)  # [1, H, W, D]
            
            resize_transform = Resize(
                spatial_size=(target_size[0], target_size[1], current_shape[2]),
                mode="bilinear"
            )
            
            ct_tensor_resampled = resize_transform(ct_tensor)
            ct_data_resampled = ct_tensor_resampled.squeeze(0).numpy()
            
            # 새 affine 행렬 계산
            scale_x = current_shape[0] / target_size[0]
            scale_y = current_shape[1] / target_size[1]
            
            new_affine = affine.copy()
            new_affine[0, 0] *= scale_x
            new_affine[1, 1] *= scale_y
            
            # 리샘플링된 CT 이미지 저장
            resampled_img = nib.Nifti1Image(ct_data_resampled, new_affine)
            nib.save(resampled_img, output_ct_file)
            print(f"  리샘플링된 CT 이미지 저장 완료: {output_ct_file}")
            
            # 2. Carotid 세그멘테이션 리샘플링 (nearest 보간)
            if os.path.exists(carotid_file):
                print("  Carotid 세그멘테이션 리샘플링 중...")
                carotid_img = nib.load(carotid_file)
                carotid_data = carotid_img.get_fdata()
                carotid_affine = carotid_img.affine
                
                carotid_tensor = torch.from_numpy(carotid_data).unsqueeze(0)  # [1, H, W, D]
                
                resize_seg_transform = Resize(
                    spatial_size=(target_size[0], target_size[1], current_shape[2]),
                    mode="nearest"
                )
                
                carotid_tensor_resampled = resize_seg_transform(carotid_tensor)
                carotid_data_resampled = carotid_tensor_resampled.squeeze(0).numpy()
                
                # 리샘플링된 Carotid 세그멘테이션 저장
                resampled_carotid = nib.Nifti1Image(carotid_data_resampled, new_affine)
                nib.save(resampled_carotid, output_carotid_file)
                print(f"  리샘플링된 Carotid 세그멘테이션 저장 완료: {output_carotid_file}")
            else:
                print(f"  경고: Carotid 세그멘테이션 파일을 찾을 수 없습니다 - {carotid_file}")
            
            # 3. Combined_Seg 세그멘테이션 리샘플링 (nearest 보간)
            if os.path.exists(combined_seg_file):
                print("  Combined_Seg 세그멘테이션 리샘플링 중...")
                combined_seg_img = nib.load(combined_seg_file)
                combined_seg_data = combined_seg_img.get_fdata()
                combined_seg_affine = combined_seg_img.affine
                
                combined_seg_tensor = torch.from_numpy(combined_seg_data).unsqueeze(0)  # [1, H, W, D]
                
                combined_seg_tensor_resampled = resize_seg_transform(combined_seg_tensor)
                combined_seg_data_resampled = combined_seg_tensor_resampled.squeeze(0).numpy()
                
                # 리샘플링된 Combined_Seg 세그멘테이션 저장
                resampled_combined_seg = nib.Nifti1Image(combined_seg_data_resampled, new_affine)
                nib.save(resampled_combined_seg, output_combined_seg_file)
                print(f"  리샘플링된 Combined_Seg 세그멘테이션 저장 완료: {output_combined_seg_file}")
            else:
                print(f"  경고: Combined_Seg 세그멘테이션 파일을 찾을 수 없습니다 - {combined_seg_file}")
            
        else:
            # 리샘플링이 필요 없으면 원본 복사
            print(f"  이미 {target_size} 크기입니다. 원본 복사 중...")
            
            # CT 이미지 복사
            nib.save(ct_img, output_ct_file)
            print(f"  CT 이미지 복사 완료: {output_ct_file}")
            
            # Carotid 세그멘테이션 복사
            if os.path.exists(carotid_file):
                carotid_img = nib.load(carotid_file)
                nib.save(carotid_img, output_carotid_file)
                print(f"  Carotid 세그멘테이션 복사 완료: {output_carotid_file}")
            else:
                print(f"  경고: Carotid 세그멘테이션 파일을 찾을 수 없습니다 - {carotid_file}")
            
            # Combined_Seg 세그멘테이션 복사
            if os.path.exists(combined_seg_file):
                combined_seg_img = nib.load(combined_seg_file)
                nib.save(combined_seg_img, output_combined_seg_file)
                print(f"  Combined_Seg 세그멘테이션 복사 완료: {output_combined_seg_file}")
            else:
                print(f"  경고: Combined_Seg 세그멘테이션 파일을 찾을 수 없습니다 - {combined_seg_file}")
    
    print("모든 케이스 처리 완료!")

def main():
    parser = argparse.ArgumentParser(description='모든 케이스의 CT 이미지와 세그멘테이션 파일을 1024x1024 크기로 리샘플링')
    parser.add_argument('--input_base_dir', type=str, required=True, 
                        help='원본 CT 이미지가 있는 기본 디렉토리 경로 (예: "data/HaN_Seg")')
    parser.add_argument('--output_base_dir', type=str, required=True, 
                        help='리샘플링된 파일을 저장할 기본 디렉토리 경로 (예: "data/Carotid_Merged")')
    parser.add_argument('--start_case', type=int, default=1, 
                        help='시작 케이스 번호 (기본값: 1)')
    parser.add_argument('--end_case', type=int, default=42, 
                        help='종료 케이스 번호 (기본값: 42)')
    
    args = parser.parse_args()
    
    resample_images(
        args.input_base_dir, 
        args.output_base_dir, 
        (1024, 1024),
        args.start_case, 
        args.end_case
    )

if __name__ == "__main__":
    main()