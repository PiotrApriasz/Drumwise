using Drumwise.Application.Entities;
using Drumwise.Infrastructure.Identity;
using Drumwise.Infrastructure.Identity.Constants;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace Drumwise.Infrastructure.Data;
using Microsoft.AspNetCore.Builder;

public static class InitializerExtensions
{
    public static async Task InitializeDatabaseAsync(this WebApplication app)
    {
        using var scope = app.Services.CreateScope();
        var initializer = scope.ServiceProvider.GetRequiredService<ApplicationDbContextInitializer>();
        await initializer.InitializeAsync();
        await initializer.SeedAsync();
    }
}

public class ApplicationDbContextInitializer
{
    private readonly ApplicationDbContext _context;
    private readonly UserManager<ApplicationUser> _userManager;
    private readonly RoleManager<IdentityRole> _roleManager;

    public ApplicationDbContextInitializer(ApplicationDbContext context,
        UserManager<ApplicationUser> userManager,
        RoleManager<IdentityRole> roleManager)
    {
        _context = context;
        _userManager = userManager;
        _roleManager = roleManager;
    }

    public async Task InitializeAsync()
    {
        try
        {
            await _context.Database.MigrateAsync();
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            throw;
        }
    }
    
    public async Task SeedAsync()
    {
        // Default roles
        var roles = new List<IdentityRole>()
        {
            new(Roles.Administrator),
            new(Roles.Student),
            new(Roles.Teacher)
        };
        
        foreach (var role in roles.Where(role => _roleManager.Roles.All(r => r.Name != role.Name)))
        {
            await _roleManager.CreateAsync(role);
        }

        // Default users
        var administrator = new ApplicationUser { UserName = "administrator@localhost", Email = "administrator@localhost" };

        if (_userManager.Users.All(u => u.UserName != administrator.UserName))
        {
            await _userManager.CreateAsync(administrator, "Administrator1!");
            if (!string.IsNullOrWhiteSpace(roles[0].Name))
            {
                await _userManager.AddToRolesAsync(administrator, new [] { roles[0].Name }!);
            }
        }

        // Default data
        // Seed, if necessary
        var guidOne = new Guid().ToString();
        var guidTwo = new Guid().ToString();
        var guidThree = new Guid().ToString();
        
        if (!_context.Lessons.Any())
        {
            _context.Lessons.Add(new Lesson
            {
                Created = DateTime.Now,
                CreatedBy = guidOne,
                LastModified = DateTime.Now,
                LastModifiedBy = guidOne,
                LessonSubject = "Quarter Notes",
                Exercise = "Play quarter notes for one hour on snare in tempo between 50 and 70"
            });
            
            await _context.SaveChangesAsync();
        }

        if (!_context.Homeworks.Any())
        {
            _context.Homeworks.Add(new Homework
            {
                Created = DateTime.Now,
                CreatedBy = guidTwo,
                LastModified = DateTime.Now,
                LastModifiedBy = guidTwo,
                AssignedTo = guidThree,
                HomeworkTitle = "Learn new song",
                Exercise = "Choose your favourite song and learn it on drums",
                Deadline = DateTime.Now.AddDays(5)
            });
            
            await _context.SaveChangesAsync();
        }

        if (!_context.UserRatings.Any())
        {
            _context.UserRatings.Add(new UserRating
            {
                Created = DateTime.Now,
                CreatedBy = guidThree,
                LastModified = DateTime.Now,
                LastModifiedBy = guidThree,
                Mark = 5,
                Comment = "The best teacher",
                AssignedTo = guidTwo
            });
            
            await _context.SaveChangesAsync();
        }
    }
}