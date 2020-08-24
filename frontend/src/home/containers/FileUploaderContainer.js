import React, { useCallback } from 'react';
import {bindActionCreators} from 'redux';
import FileUploader from '../components/FileUploader'
import { select, upload } from '../modules/FileUploader';
import { connect, useSelector, useDispatch } from 'react-redux';

const FileUploaderContainer = () => {
  const {fileName, result} = useSelector(({FileUploader}) => ({
    fileName: FileUploader.name,    
    result: FileUploader.result
  }));
  const dispatch = useDispatch();
  const onChangeName = useCallback(name => dispatch(select(name)), [dispatch]);
  const onClickUpload = useCallback(() => dispatch(upload()), [dispatch]);
  
  return (
    <FileUploader 
      name={fileName}
      result={result}
      onChange={onChangeName}
      onClickUpload={onClickUpload}      
    />
  );
};

export default React.memo(FileUploaderContainer);