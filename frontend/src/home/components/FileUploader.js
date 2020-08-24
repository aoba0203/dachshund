import React from 'react';
import Typography from '@material-ui/core/Typography'

const FileUploader = ({
  name,  
  result,
  onChange,
  onClickUpload
}) => {
  const onClick=() => {    
    onChange('');
    onClickUpload(name);
  };
  const handleFileInput = e => onChange(e.target.files[0]);
  return (
    <div>
      <input type="file" name="file" onChange={handleFileInput}/>
      <button type="button" onClick={onClick} />
      <h1>{result}</h1>
    </div>
  );
};

export default FileUploader;
