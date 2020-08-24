import {createAction, handleActions} from 'redux-actions';
import Counter from '../components/Counter';
const INCREASE = 'counter/INCREASE';
const DECREASE = 'counter/DECREASE';
const CHANGE_INPUT = 'counter/CHANGE_INPUT';
const CHANGE_NAME = 'counter/CHANGE_NAME';

export const increase = createAction(INCREASE);
export const decrease = createAction(DECREASE);
export const changeInput = createAction(CHANGE_INPUT, input => input);
export const changeName = createAction(CHANGE_NAME, name => name);

const initialState = {
  number: 0,
  name: 'Counter',
  inputName: ''
};

const counter = handleActions(
  {
    [INCREASE]: (state, action) => ({
      ...state,
      number: state.number + 1
    }),
    [DECREASE]: (state, action) => ({
      ...state,
      number: state.number -1
    }),
    [CHANGE_INPUT]: (state, {payload: input}) =>({
      ...state,
      inputName: input
    }),
    [CHANGE_NAME]: (state, {payload: name}) => ({
      ...state,
      name: name
    })
  },
  initialState
);

export default counter;