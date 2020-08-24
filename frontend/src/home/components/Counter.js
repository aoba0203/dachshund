import React from 'react';

const Counter = ({
  name,
  number,
  inputName, 
  onIncrease, 
  onDecrease,
  onChange,
  onChangeName
}) => {
  const onClick=() => {
    onChangeName(inputName);
    onChange('');
  };
  const onInputChange = e => onChange(e.target.value);
  return (
    <div>
      <h1>{name}</h1>
      <div>
        <h1>{number}</h1>
        <button onClick={onIncrease}>+1</button>
        <button onClick={onDecrease}>-1</button>
      </div>
      <input
        onChange={onInputChange} 
        value={inputName}
        placeholder="Input Counter name"
      ></input>
      <button onClick={onClick}>Register</button>
    </div>
  );
};

export default Counter;
