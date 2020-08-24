import React, { useCallback } from 'react';
import {bindActionCreators} from 'redux';
import Counter from '../components/Counter';
import { increase, decrease, changeInput, changeName } from '../modules/Counter';
import { connect, useSelector, useDispatch } from 'react-redux';

const CounterContainer = () => {
  const {counterNum, name, inputName} = useSelector(({counter}) => ({
    counterNum: counter.number,
    name: counter.name,
    inputName: counter.inputName
  }));
  const dispatch = useDispatch();
  const onIncrease = useCallback(() => dispatch(increase()), [dispatch]);
  const onDecrease = useCallback(() => dispatch(decrease()), [dispatch]);
  const onInputChange = useCallback(input => dispatch(changeInput(input)), [dispatch]);
  const onChangeName = useCallback(name => dispatch(changeName(name)), [dispatch]);
  return (
    <Counter 
      name={name}
      number={counterNum}
      inputName={inputName} 
      onIncrease={onIncrease} 
      onDecrease={onDecrease} 
      onChange={onInputChange}
      onChangeName={onChangeName}
    />
  );
};

export default React.memo(CounterContainer);