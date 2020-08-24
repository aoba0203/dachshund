import React, {Component} from 'react';
import Project from './Project';
import Table from '@material-ui/core/Table';
import TableHead from '@material-ui/core/TableHead';
import TableBody from '@material-ui/core/TableBody';
import TableRow from '@material-ui/core/TableRow';
import TableCell from '@material-ui/core/TableCell';

const rows = [
  {'id':1,
    'project_name': 'project_1',
    'problem_type': 'Regression',
    'metrics_name': 'AbsolutePercentError',
    'score': 457.4,
    'created': '2020-08-19T10:08:28'},
  {'id':2,
    'project_name': 'project_2',
    'problem_type': 'Regression',
    'metrics_name': 'AbsolutePercentError',
    'score': 424.4,
    'created': '2020-08-19T14:49:25'},
  {'id':3,
    'project_name': 'project_3',
    'problem_type': 'Regression',
    'metrics_name': 'AbsolutePercentError',
    'score': 414.4,
    'created': '2020-08-18T14:49:25'},
];

class ProjectHome extends Component{
  render(){
    console.log('ProjectHome') 
    return (
      <div>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Metrics</TableCell>
              <TableCell>Score</TableCell>
              <TableCell>Created</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map(row => {
              return <Project id={row.id} project_name={row.project_name} problem_type={row.problem_type} metrics_name={row.metrics_name} score={row.score} created={row.created}/>
            })}
          </TableBody>
        </Table>
      </div>
    )
  }
}

export default ProjectHome;