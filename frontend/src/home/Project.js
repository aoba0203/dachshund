import React from 'react';
import TableRow from '@material-ui/core/TableRow';
import TableCell from '@material-ui/core/TableCell';

class Project extends React.Component{  
  handleClick(){
    console.log('click: ', this.props.id)
  }
  render(){
    return (
      <TableRow hover={true} onClick={ this.handleClick.bind(this) }>
        <TableCell>{this.props.project_name}</TableCell>
        <TableCell>{this.props.problem_type}</TableCell>
        <TableCell>{this.props.metrics_name}</TableCell>
        <TableCell>{this.props.score}</TableCell>
        <TableCell>{this.props.created}</TableCell>
      </TableRow>
    )
  }
}

export default Project;