import { combineReducers } from 'redux';
import counter from './Counter';
import fileUploader from './FileUploader'

const rootReducer = combineReducers({
  counter,
  fileUploader
});

export default rootReducer;
