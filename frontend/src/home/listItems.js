import React from 'react';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';
import LIstSubheader from '@material-ui/core/ListSubheader';
import DashboardIcon from '@material-ui/icons/Dashboard';
import ShoppingCartIcon from '@material-ui/icons/ShoppingCart';
import PeopleIcon from '@material-ui/icons/People';
import BarChartIcon from '@material-ui/icons/BarChart';
import LayersIcon from '@material-ui/icons/Layers';
import AppsIcon from '@material-ui/icons/Apps';
import AssignmentIcon from '@material-ui/icons/Assignment';
import { Link as RouterLink } from 'react-router-dom';
import Link from '@material-ui/core/Link';

export const mainListItems = (
  <div>
    <Link component={RouterLink} to="/">
      <ListItem button>      
        <ListItemIcon>
          {/* <DashboardIcon /> */}
          <AppsIcon />
        </ListItemIcon>
        <ListItemText primary='Project list' />      
      </ListItem>
    </Link>
    <Link component={RouterLink} to="/detail">
      <ListItem button>
        <ListItemIcon>
          <ShoppingCartIcon />
        </ListItemIcon>
        <ListItemText primary='Orders' />      
      </ListItem>
    </Link>
  </div>
);

export const secondaryListItems = (
  <div>
    <LIstSubheader inset> Saved reposts</LIstSubheader>
    <ListItem button>
      <ListItemIcon>
        <AssignmentIcon />
      </ListItemIcon>
      <ListItemText primary='Current month' />
    </ListItem>
    <ListItem button>
      <ListItemIcon>
        <AssignmentIcon />
      </ListItemIcon>
      <ListItemText primary='Last quarter' />
    </ListItem>
    <ListItem button>
      <ListItemIcon>
        <AssignmentIcon />
      </ListItemIcon>
      <ListItemText primary='Year-end sale' />
    </ListItem>
  </div>
)