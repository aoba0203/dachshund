import {createAction, handleActions} from 'redux-actions';
import axios from 'axios'

const FILE_SELECT = 'upload/FILE_SELECT';
const FILE_UPLOAD = 'upload/FILE_UPLOAD';

export const select = createAction(FILE_SELECT, name => name);
export const upload = createAction(FILE_UPLOAD);

const initialState = {
  fileName: 'file name'
};

function uploadFile(){
  console.log('uploadFile: ', this.fileName)
}

const fileUploader = handleActions(
  {
    [FILE_SELECT]: (state, {payload: name}) => ({
      ...state,
      fileName: name
    }),
    [FILE_UPLOAD]: (state, action) => ({
      ...state,
      uploadFile
    })
  },
  initialState
);

export default fileUploader;