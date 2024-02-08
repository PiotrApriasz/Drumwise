using System.Reflection;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Entities;
using Microsoft.EntityFrameworkCore;

namespace Drumwise.Infrastructure.Data;

public class ApplicationDbContext(DbContextOptions<ApplicationDbContext> options)
    : DbContext(options), IApplicationDbContext
{
    public DbSet<Homework> Homeworks => Set<Homework>();
    public DbSet<Lesson> Lessons => Set<Lesson>();
    public DbSet<UserRating> UserRatings => Set<UserRating>();
    
    protected override void OnModelCreating(ModelBuilder builder)
    {
        builder.ApplyConfigurationsFromAssembly(Assembly.GetExecutingAssembly());
        base.OnModelCreating(builder);
    }
}