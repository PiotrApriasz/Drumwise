using System.Reflection;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Entities;
using Drumwise.Infrastructure.Identity;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;

namespace Drumwise.Infrastructure.Data;

public class AppIdentityDbContext(DbContextOptions<AppIdentityDbContext> options)
    : IdentityDbContext<ApplicationUser>(options), IApplicationDbContext
{
}