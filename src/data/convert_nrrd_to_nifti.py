import autorootcwd
import os
import glob
import argparse
import numpy as np
import nibabel as nib
import nrrd

def convert_nrrd_to_nifti(input_dir, output_dir):
    """
    NRRD 파일들을 NIFTI 형식으로 변환합니다.
    
    Args:
        input_dir: NRRD 파일이 있는 디렉토리 경로
        output_dir: 변환된 NIFTI 파일을 저장할 디렉토리 경로
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 입력 디렉토리에서 모든 NRRD 파일 찾기
    nrrd_files = glob.glob(os.path.join(input_dir, "*.nrrd"))
    
    print(f"총 {len(nrrd_files)}개의 NRRD 파일을 찾았습니다.")
    
    for nrrd_file in nrrd_files:
        filename = os.path.basename(nrrd_file)
        basename = os.path.splitext(filename)[0]
        output_filename = basename + ".nii.gz"
        output_filepath = os.path.join(output_dir, output_filename)
        
        print(f"변환 중: {filename} -> {output_filename}")
        
        try:
            # NRRD 파일 로드
            data, header = nrrd.read(nrrd_file)
            
            # NRRD에서 affine 행렬 구성
            space_directions = header.get('space directions')
            space_origin = header.get('space origin', np.zeros(3))
            
            if space_directions is not None:
                # space_directions를 affine 행렬로 변환
                affine = np.eye(4)
                affine[:3, :3] = space_directions
                affine[:3, 3] = space_origin
            else:
                # space_directions가 없는 경우 기본 affine 사용
                affine = np.eye(4)
            
            # NIFTI 이미지 생성 및 저장
            nifti_img = nib.Nifti1Image(data, affine)
            nib.save(nifti_img, output_filepath)
            
            print(f"성공적으로 변환됨: {output_filepath}")
            
        except Exception as e:
            print(f"오류 발생: {filename} - {str(e)}")

def convert_all_cases(input_base_dir, output_base_dir, start_case=1, end_case=42):
    """
    모든 case 폴더에 대해 NRRD 파일을 NIFTI로 변환합니다.
    """
    # 출력 기본 디렉토리가 없으면 생성
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 각 케이스 처리
    for case_num in range(start_case, end_case + 1):
        case_id = f"case_{case_num:02d}"  # 예: case_01, case_02, ...
        
        input_dir = os.path.join(input_base_dir, case_id)
        output_dir = os.path.join(output_base_dir, case_id)
        
        # 입력 디렉토리가 존재하는지 확인
        if not os.path.exists(input_dir):
            print(f"경고: 입력 디렉토리가 존재하지 않습니다 - {input_dir}")
            continue
        
        print(f"\n=== 처리 중: {case_id} ===")
        # 단일 케이스 변환 함수 직접 호출
        convert_nrrd_to_nifti(input_dir, output_dir)
        print(f"{case_id} 변환 완료\n")
    
    print("모든 케이스 변환 완료!")

def main():
    parser = argparse.ArgumentParser(description='모든 케이스에 대해 NRRD 파일을 NIFTI로 변환')
    parser.add_argument('--input_base_dir', type=str, required=True, 
                        help='NRRD 파일이 있는 기본 디렉토리 경로 (예: "data/HaN_Seg")')
    parser.add_argument('--output_base_dir', type=str, required=True, 
                        help='변환된 NIFTI 파일을 저장할 기본 디렉토리 경로 (예: "data/converted_nifti")')
    parser.add_argument('--start_case', type=int, default=1, 
                        help='시작 케이스 번호 (기본값: 1)')
    parser.add_argument('--end_case', type=int, default=42, 
                        help='종료 케이스 번호 (기본값: 42)')
    
    args = parser.parse_args()
    
    convert_all_cases(
        args.input_base_dir, 
        args.output_base_dir, 
        args.start_case, 
        args.end_case
    )

if __name__ == "__main__":
    main()